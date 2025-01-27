# Copyright (c) Meta Platforms, Inc. and affiliates.

import contextlib
import json
import logging
import os
from copy import deepcopy
from dataclasses import dataclass, field
from functools import partial
from multiprocessing import Event, Process, Queue
from multiprocessing.synchronize import Event as EventClass
from pathlib import Path
from queue import Empty, Full
from typing import Any, Dict, Iterator, Optional, TypedDict

import numpy as np

from lingua.tokenizer import TokenizerArgs, build_tokenizer

logger = logging.getLogger()

"""
This file contains all code necessary for text data loading from preshuffled jsonl chunks.
For example if given the following files with a world size of 8 

/path/to/arxiv:
arxiv.chunk.00.jsonl (Contains many lines of {"text":...} or {"content":...})
arxiv.chunk.01.jsonl
arxiv.chunk.02.jsonl
arxiv.chunk.03.jsonl

/path/to/wikipedia:
wikipedia.chunk.00.jsonl
wikipedia.chunk.01.jsonl
wikipedia.chunk.02.jsonl
wikipedia.chunk.03.jsonl

Step (1) => infinite_block_jsonl_iterator
2 workers will read each jsonl chunk (world_size = 8 distributed over 4 workers) from each source.
Each worker will read 1 line and skip the next, therefore workers on the same file read in an interleaved manner.

Step (2) => multi_choice_iterator
At every iteration, a source is sampled randomly given some weights

Step (3) => tokenizer and pack_tokens
Reads sequences until reaching seq_len tokens and yields a numpy array of shape (seq_len, n_views)

Step (4) => prefetch_data_loader
Prefetches batches in advance and shuffles them to reduce correlation, yields a numpy array of shape (batch_size, seq_len, n_views)

This create a nested iterator structure where each iterator is responsible for a specific task:
    [ [ [ [ [ (1) read document ] -> (2) sample source ] -> (3) tokenize ] -> (4) tokenize and build sequence of fixed seq_len ] -> (5) prefetch batches ]

Each iterator returns a tuple (output, state) where state contains all the info necessary to resume from the last output.

build_mixed_token_packing_dataloader creates the states and return an iterator that does everything above

build_seperate_token_packing_dataloader does the same thing but swaps step 2 and 3 

Both can be called with a resume_state to resume from any given position deterministically
"""

TRAIN_DATA_FILE_PATTERN = "*.chunk.*.jsonl"


class JSONLState(TypedDict):
    """Represents the current state of a JSON line reader.

    Attributes:
        content (Dict): The JSON content of the line.
        file_path (str): The path to the JSONL file.
        position (int): The file position after reading the line (in bytes).
        window (int): The window size used for iteration.
        offset (int): The offset used for iteration.
        current_iter (Optional[int]): Number of iterations over the jsonl file (for infinite iteration).
    """

    file_path: str
    position: int
    block_size: int
    offset: int
    current_iter: int


class MultiChoiceState(TypedDict):
    """Represents the current state of a Multi choice iterator.

    Attributes:
        root_dir: path to dataset root directory
        sources Dict[str, float]: Dict from subdirectory to the weight used for sampling
        source_states: Dict[str, Any] Dict from source to iterator state
        rng_state: dict numpy bit generator state used to resume rng
    """

    root_dir: str
    sources: Dict[str, float]
    source_to_state: Dict[str, Any]
    rng_state: Dict[str, Any]


class TokenizerState(TypedDict):
    it_state: Any
    name: str
    add_bos: bool
    add_eos: bool
    path: Optional[str]


class PackTokensState(TypedDict):
    """Represents the current state of a packing iterator.

    Attributes:
        start_token: int index to start reading from in the current sequence
        output_seq_len: int Length of sequences to output
        n_views: dict int Number of views to output. Each view is the same sequence but shifted by 1 from the previous
    """

    start_token: int
    it_state: Any
    output_seq_len: int
    n_views: int
    seq_len: int


class PrefetchState(TypedDict):
    """Represents the current state of a prefetching iterator.

    Attributes:
        prefetch_buffer: numpy array to store prefetched data
        seq_idx: int index of the current sequence to resume from
        rng_state: dict numpy bit generator state used to resume rng
    """

    it_state: Any
    seq_idx: int
    rng_state: Dict[str, Any]
    prefetch_size: int
    batch_size: int


def read_jsonl(
    file_path: str,
    position: int,
    block_size: int,
    offset: int,
    current_iter: int,
):
    """Iterates over a JSON Lines file, yielding a line every `block_size` lines with an offset

    Example : If block_size = 3, offset = 1, iterator will yield lines 1 4 7 10 ...
    Example : If block_size = 2, offset = 0, iterator will yield lines 0 2 4 6 ...

    Args:
        file_path (str): Path to the JSONL file.
        position (int): The file position (in bytes) from which to start reading.
        block_size (int): The number of lines to skip between yields
        offset (int): The initial number of lines skiped

    Yields:
        JSONLState: Represents the state of each line read according to window and offset.
    """
    if (offset < 0) or (offset >= block_size):
        raise RuntimeError(f"JSONL iterator offset value is invalid")
    # We assume the start position is either 0 or given by the last line yielded
    # Therefore the current line is right after the offset (modulo block_size)
    current_line = offset + 1 if position > 0 else 0

    state = JSONLState(
        file_path=file_path,
        position=position,
        block_size=block_size,
        offset=offset,
        current_iter=current_iter,
    )
    with open(file_path, "r", encoding="utf-8") as file:
        file.seek(position)
        while line := file.readline():
            current_line += 1
            if (current_line - 1) % block_size == offset:
                # We return state that will allow resuming from this position
                # We update state for next position
                state = JSONLState(
                    file_path=file_path,
                    position=file.tell(),
                    block_size=block_size,
                    offset=offset,
                    current_iter=current_iter,
                )
                yield json.loads(line), state


def loop_on_jsonl(
    file_path: str,
    position: int,
    block_size: int,
    offset: int,
    current_iter: int,
):
    """Makes the block jsonl iterator infinite and updates n_iter counter"""
    try:
        while True:
            it = read_jsonl(file_path, position, block_size, offset, current_iter)
            for content, jsonl_state in it:
                yield content, jsonl_state
            current_iter += 1
            position = 0
    finally:
        it.close()


def tokenize(
    iterator: Iterator,
    add_bos: bool,
    add_eos: bool,
    tokenizer_type: str,
    tokenizer_path: Optional[str] = None,
):
    """
    Tokenizes text from an iterator of content-state pairs using a specified tokenizer.

    Parameters:
    - iterator: An iterable of (content, state) pairs where content is a dict with a 'text' or 'content' key.
    - tokenizer: Tokenizer object with an `encode` method to convert text to tokens, supporting `add_bos` and `add_eos`.
    - add_bos (bool): Flag to add a beginning-of-sequence token.
    - add_eos (bool): Flag to add an end-of-sequence token.

    Yields:
    - (tokens, state) pairs, where `tokens` is a list of tokenized text, and `state` is the original state from the iterator.
    """
    tokenizer = build_tokenizer(name=tokenizer_type, path=tokenizer_path)
    for content, state in iterator:
        assert (
            "text" in content or "content" in content
        ), "JSON line must contain either text or content key"
        content_key = "text" if ("text" in content) else "content"
        text = content[content_key]
        tokens = tokenizer.encode(text, add_bos=add_bos, add_eos=add_eos)
        yield tokens, TokenizerState(
            it_state=state,
            add_bos=add_bos,
            add_eos=add_eos,
            name=tokenizer_type,
            path=tokenizer_path,
        )


def choose_source(
    source_to_iterator: Dict[str, Iterator],
    source_to_state: Dict[str, Any],
    root_dir: str,
    sources: Dict[str, float],
    rng_state: Dict[str, Any],
):
    """
    Iterates over multiple data sources, selecting sequences based on weighted random choice.

    Parameters:
    - source_to_iterator (Dict[str, Iterator]): Dict from source paths to their iterators.
    - source_to_state (Dict[str, State]): Initial state for each source, allowing state tracking.
    - root_dir str: Root dir of data sources
    - sources Dict[str, float]: Dict from subdirectory to the weight used for sampling
    - rng_state (dict): State of the random number generator for reproducibility.

    Yields:
    - Tuple of (seq, multi_choice_state) where `seq` is the next sequence from the chosen source,
    and `multi_choice_state` includes the current state of all sources and the RNG.

    This function ensures that sequences are chosen from the provided sources based on the specified weights,
    maintaining state information for each source and the RNG to allow for reproducible iteration.
    """
    n_sources = len(sources)
    possible_sources = list(sources.keys())
    weights = list(sources.values())
    # We create the rng and set its state
    rng = np.random.default_rng()
    rng.bit_generator.state = rng_state
    while True:
        # We save the rng state before sampling to be able to yield the same sequence on reload
        norm_weights = np.array(weights) / np.array(weights).sum()
        source_choice = possible_sources[rng.choice(n_sources, p=norm_weights)]
        seq, state = next(source_to_iterator[source_choice])
        source_to_state = {**source_to_state, source_choice: state}
        # We update the corresponding source state
        multi_choice_state = MultiChoiceState(
            root_dir=root_dir,
            sources=sources,
            source_to_state=source_to_state,
            rng_state=rng.bit_generator.state,
        )
        yield seq, multi_choice_state


def get_empty_buffer_state(
    start_token,
    states,
):
    """
    Calculates the state to resume iteration after the buffer is cleared.

    This function determines the starting point for resuming iteration by rewinding `n_views` from the `end_token`.
    It handles cases where the rewind goes beyond the current sequence, adjusting the starting sequence and token index accordingly.
    """
    # We rewind n_views
    # This index can be negative if we go beyond the current sample
    # In that case we go back to find which sequence to start from
    # And the correct token index to start from
    seq_to_resume_from = -1
    while start_token < 0:
        seq_to_resume_from -= 1
        start_token += states[seq_to_resume_from]["seq_len"]
    resume_state = deepcopy(states[seq_to_resume_from])
    resume_state["start_token"] = start_token
    # When resuming, the iterator will then correctly fill the buffer
    del states[:seq_to_resume_from]
    if "seq_len" in resume_state:
        del resume_state["seq_len"]

    return resume_state


def pack_tokens(
    iterator: Iterator,
    empty_buffer_state: PackTokensState,
):
    """
    Iterates over tokens, packing them into chunks.

    This function aggregates tokens into a buffer and yields fixed-size chunks with dimensions `(output_seq_len, n_views)`,
    where each column represents shifted sequences of tokens. It ensures continuity in token sequences across chunks,
    preventing boundary effects and maintaining consistency regardless of `n_views`.

    Parameters:
    - iterator: An iterator that yields pairs of (tokens, state), where tokens is a 1D sequence of tokens and state contains all necessary information to resume iterator from current position.
    - it_state: State of the iterator currently.
    - start_token (int): The index of the first token to start reading from for the first sequence.
    - output_seq_len (int): The length of the output sequences to be generated.
    - n_views (int): The number of shifted views to include in each output chunk.

    Yields:
    - numpy.ndarray: An array of shape `(output_seq_len, n_views)` containing the packed tokens.
    - PackTokensState: The state required to resume packing tokens from where the last returned chunk.

    The function handles the complexity of determining the correct state for resuming iteration after the buffer is cleared, ensuring seamless continuation of token sequences.
    """
    buffer = []
    states = []
    output_seq_len = empty_buffer_state["output_seq_len"]
    n_views = empty_buffer_state["n_views"]
    start_token = empty_buffer_state["start_token"]
    previous_state = empty_buffer_state["it_state"]
    buffer_size = output_seq_len + n_views - 1
    for i, (tokens, state) in enumerate(iterator):
        end_token = start_token
        sample_is_read = False
        while not sample_is_read:
            assert start_token < len(
                tokens
            ), f"Start token index {start_token} bigger than sequence {len(tokens)}"
            free_space = buffer_size - len(buffer)
            seq_len = min(free_space, len(tokens) - start_token)
            end_token = start_token + seq_len
            buffer.extend(tokens[start_token:end_token])
            start_token = end_token

            states.append(
                PackTokensState(
                    start_token=start_token,
                    seq_len=seq_len,
                    it_state=previous_state,
                    output_seq_len=output_seq_len,
                    n_views=n_views,
                )
            )
            assert len(buffer) <= buffer_size, "Buffer overflow"

            if len(buffer) == buffer_size:
                out = np.array(buffer)
                assert out.ndim == 1, "Iterator should return 1D sequences"
                out = np.lib.stride_tricks.sliding_window_view(
                    out, n_views, axis=0
                )  # (output_seq_len, n_views)

                # We rewind by n_views to account for the last tokens not having their targets
                rewinded_idx = start_token - (n_views - 1)
                empty_buffer_state = get_empty_buffer_state(rewinded_idx, states)
                buffer = buffer[output_seq_len:]
                assert len(buffer) == (n_views - 1)

                yield out, empty_buffer_state

            if start_token == len(tokens):
                start_token = 0
                sample_is_read = True
                previous_state = state


def batch_and_shuffle_prefetched_sequences(
    data_loader: Iterator,
    batch_size: int,
    prefetch_size: int,
    seq_len: int,
    n_views: int,
    state: PrefetchState,
):
    """
    Prepare batch in advance and shuffle them to reduce correlation inside batches (for ex when very long document is encountered).

    This function aggregates batches into a buffer and yields fixed-size batch size and seqlen with dimensions `(batch_size, seqlen, n_views)`,

    It uses a prefetch buffer to store batches in advance and shuffles them, the prefetch buffer is similar to `reservoir sampling`,
    but by block to preserve a smooth, easy and deterministic reloading. To ensure more uniform sequence sampling -> prefetch_size * batch_size * seq_len >> max_document_seqlength.

    Parameters:
    - iterator: An iterator that yields pairs of (sequence, state), where is a random sequence sampled from a corpus (as done by pack_tokens for example).
    - batch_size: The desired batch size.
    - prefetch_size: The number of batches to prefetch in advance.
    - seq_len (int): The length of the output sequences to be generated.
    - n_views (int): The number of shifted views to include in each output chunk.

    Yields:
    - numpy.ndarray: An array of shape `(batch_size, seq_len, n_views)` containing the packed tokens.
    - PrefetchState: The state required to resume prefetched batch. Contains also the internal of iterator.
    """
    prefetch_buffer = -1 * np.ones((prefetch_size * batch_size, seq_len, n_views), dtype=int)
    rng = np.random.default_rng()
    rng.bit_generator.state = state["rng_state"]

    # Rewind the iterator to the correct position by skipping seq_idx sequences to roll the buffer accordingly
    seq_idx = state["seq_idx"]
    assert (
        seq_idx >= 0 and seq_idx < prefetch_size
    ), "Prefetch state seq_idx should be in 0 <= seq_idx < prefetch_size."

    _rng_state = state["rng_state"]
    _it_state = state["it_state"]

    for i in range(prefetch_size * batch_size):
        prefetch_buffer[i], next_it_state = next(data_loader)
    rng.shuffle(prefetch_buffer, axis=0)
    for i in range(seq_idx * batch_size):
        prefetch_buffer[i], _ = next(data_loader)

    idx = seq_idx
    while True:
        if idx == prefetch_size - 1:
            _it_state = next_it_state
            _rng_state = rng.bit_generator.state

        state = PrefetchState(
            it_state=_it_state,
            seq_idx=(idx + 1) % prefetch_size,
            rng_state=_rng_state,
            batch_size=batch_size,
            prefetch_size=prefetch_size,
        )

        yield prefetch_buffer[idx * batch_size : (idx + 1) * batch_size].copy(), state

        for i in range(batch_size):
            prefetch_buffer[idx * batch_size + i], pack_state = next(data_loader)

        if idx == prefetch_size - 1:
            next_it_state = pack_state
            rng.shuffle(prefetch_buffer, axis=0)

        idx = (idx + 1) % prefetch_size


def find_and_sanitize_chunks(
    dataset_path: str, world_size: int, file_pattern: str = TRAIN_DATA_FILE_PATTERN
):
    dataset_chunks = [str(p) for p in Path(dataset_path).glob(file_pattern)]
    n_chunks = len(dataset_chunks)

    if n_chunks > world_size:
        n_discard = n_chunks - world_size
        dataset_chunks = dataset_chunks[:world_size]
    else:
        assert world_size % n_chunks == 0, "World size should be a multiple of number of chunks"

    assert n_chunks > 0, f"No valid chunks in {dataset_path}"

    return dataset_chunks


def distribute_data_to_rank(dataset_path: str, rank: int, world_size: int, file_pattern: str):
    """
    Distributes the chunk files in a dataset path to each worker.
    If world_size is smaller than the number of chunks, the extra chunks are discarded.
    Otherwise, world_size is assumed to be a multiple of number of chunks.
    In that case there are world_size//nb_chunks workers on each chunk file, reading with different offsets.
    """
    dataset_chunks = find_and_sanitize_chunks(dataset_path, world_size, file_pattern)
    n_ranks_per_chunk = world_size // len(dataset_chunks)
    rank_to_jsonl_iterator_params = []
    for chunk_path in dataset_chunks:
        for i in range(n_ranks_per_chunk):
            rank_to_jsonl_iterator_params.append(
                JSONLState(
                    file_path=chunk_path,
                    position=0,
                    block_size=n_ranks_per_chunk,
                    offset=i,
                    current_iter=0,
                )
            )

    return rank_to_jsonl_iterator_params[rank]


def init_choice_state(
    root_dir: str,
    sources: Dict[str, float],
    seed: int,
    rank: int,
    world_size: int,
    file_pattern: str,
):
    data_path_to_jsonl_state = dict()
    for dataset_path in sources:
        jsonl_state = distribute_data_to_rank(
            os.path.join(root_dir, dataset_path), rank, world_size, file_pattern
        )
        data_path_to_jsonl_state[dataset_path] = jsonl_state

    multi_rng_state = np.random.default_rng((seed, rank, world_size)).bit_generator.state

    multi_choice_state = MultiChoiceState(
        root_dir=root_dir,
        sources=sources,
        source_to_state=data_path_to_jsonl_state,
        rng_state=multi_rng_state,
    )
    return multi_choice_state


def init_state(
    root_dir: str,
    sources: Dict[str, float],
    batch_size: int,
    prefetch_size: int,
    seq_len: int,
    n_views: int,
    seed: int,
    rank: int,
    world_size: int,
    add_bos: bool,
    add_eos: bool,
    tokenizer_name: str,
    tokenizer_path: Optional[str] = None,
    file_pattern: str = TRAIN_DATA_FILE_PATTERN,
):
    multi_choice_state = init_choice_state(
        root_dir=root_dir,
        sources=sources,
        seed=seed,
        rank=rank,
        world_size=world_size,
        file_pattern=file_pattern,
    )
    tokenizer_state = TokenizerState(
        it_state=multi_choice_state,
        add_bos=add_bos,
        add_eos=add_eos,
        name=tokenizer_name,
        path=tokenizer_path,
    )
    pack_state = PackTokensState(
        start_token=0,
        it_state=tokenizer_state,
        output_seq_len=seq_len,
        n_views=n_views,
        seq_len=0,
    )

    prefetch_rng_state = np.random.default_rng((seed + 1, rank, world_size)).bit_generator.state

    return PrefetchState(
        it_state=pack_state,
        seq_idx=0,
        rng_state=prefetch_rng_state,
        batch_size=batch_size,
        prefetch_size=prefetch_size,
    )


def setup_sources(multi_state):
    path_to_iter = dict()
    for source in multi_state["sources"]:
        jsonl_state = multi_state["source_to_state"][source]
        path_to_iter[source] = loop_on_jsonl(
            jsonl_state["file_path"],
            jsonl_state["position"],
            jsonl_state["block_size"],
            jsonl_state["offset"],
            jsonl_state["current_iter"],
        )

    return path_to_iter


@contextlib.contextmanager
def build_dataloader(
    state: PrefetchState,
):
    pack_state = state["it_state"]
    tokenizer_state = pack_state["it_state"]
    multi_state = tokenizer_state["it_state"]

    path_to_iter = setup_sources(multi_state)
    data_it = choose_source(
        source_to_iterator=path_to_iter,
        source_to_state=multi_state["source_to_state"],
        root_dir=multi_state["root_dir"],
        sources=multi_state["sources"],
        rng_state=multi_state["rng_state"],
    )
    data_it = tokenize(
        data_it,
        tokenizer_state["add_bos"],
        tokenizer_state["add_eos"],
        tokenizer_state["name"],
        tokenizer_state["path"],
    )

    data_it = pack_tokens(
        data_it,
        pack_state,
    )

    data_it = batch_and_shuffle_prefetched_sequences(
        data_loader=data_it,
        seq_len=pack_state["output_seq_len"],
        n_views=pack_state["n_views"],
        batch_size=state["batch_size"],
        prefetch_size=state["prefetch_size"],
        state=state,
    )
    yield data_it
    for it in path_to_iter.values():
        it.close()
    data_it.close()


def feed_buffer(queue: Queue, stop_event: EventClass, iterator_builder):
    """
    Producer function to fetch data from an iterable dataset and put it into a queue.
    Incorporates timeout management to avoid hanging on queue.put() when the queue is full.
    """
    with iterator_builder() as iterator:
        for item in iterator:
            while not stop_event.is_set():
                try:
                    queue.put(
                        item, timeout=0.1
                    )  # Attempts to put item into the queue with a timeout
                    break  # On successful put, breaks out of the while loop
                except Full:
                    pass
            if stop_event.is_set():
                break


def consume_buffer(producer: Process, queue: Queue):
    """
    Consumer function to process items from the queue.
    Handles cases where the queue might be empty by implementing timeouts on queue.get().
    """
    while producer.exitcode is None:
        try:
            item = queue.get(timeout=0.1)  # Tries to get an item from the queue with a timeout
            yield item
        except Empty:
            pass

    raise RuntimeError("Data loader quit unexpectedly, real error has been raised previously")


@contextlib.contextmanager
def async_iterator(buffer_size: int, iterator_builder):
    """
    Context manager to setup and manage asynchronous iteration with producer-consumer model.
    """
    queue = Queue(maxsize=buffer_size)
    stop_event = Event()
    producer = Process(target=feed_buffer, args=(queue, stop_event, iterator_builder))
    logger.info("Async dataloader started")
    producer.start()

    consumer = consume_buffer(producer, queue)
    try:
        yield consumer
    finally:
        stop_event.set()  # Ensures the stop event is signaled
        consumer.close()
        producer.join(timeout=0.2)  # Waits for the producer to finish
        if producer.exitcode is None:
            logger.info(f"Killing async data process {producer.pid} ...")
            producer.kill()
        else:
            logger.info(f"Async data process {producer.pid} exited with code {producer.exitcode}")
        logger.info("Async dataloader cleaned up")


@dataclass
class DataArgs:
    root_dir: Optional[str] = None
    sources: Dict[str, float] = field(default_factory=dict)
    batch_size: int = 2
    seq_len: int = 2048
    n_views: int = 2
    seed: int = 42
    add_bos: bool = True
    add_eos: bool = True
    load_async: bool = True
    prefetch_size: int = 64
    tokenizer: TokenizerArgs = field(default_factory=TokenizerArgs)


def init_dataloader_state_from_args(
    args: DataArgs,
    rank: int,
    world_size: int,
):
    return init_state(
        root_dir=args.root_dir,
        sources=args.sources,
        seq_len=args.seq_len,
        batch_size=args.batch_size,
        prefetch_size=args.prefetch_size,
        n_views=args.n_views,
        seed=args.seed,
        rank=rank,
        world_size=world_size,
        tokenizer_name=args.tokenizer.name,
        tokenizer_path=args.tokenizer.path,
        add_bos=args.add_bos,
        add_eos=args.add_eos,
    )


def build_dataloader_from_args(
    args: DataArgs,
    state: Optional[PrefetchState] = None,
):
    data_builder = partial(build_dataloader, state)
    if args.load_async:
        return async_iterator(args.prefetch_size, data_builder)
    else:
        return data_builder()
