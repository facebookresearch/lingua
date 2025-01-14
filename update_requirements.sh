echo "Updating requirements_lock.txt" && \
bazel run //:requirements.update && \
echo "Updating pypi_pkgs_lock.bzl" && \
bazel query 'kind("alias rule", deps(@pip//...))' --output=label_kind | \
  grep ":pkg" | \
  grep -oP "@.+" | \
  sed -e "s/.*/'&',/" -e '1s/^/PIPY_PKGS = [/' -e '$s/,$/]/' > pypi_pkgs_lock.bzl && \
echo "Updating gazelle_python.yaml" && \
bazel run //:gazelle_python_manifest.update
