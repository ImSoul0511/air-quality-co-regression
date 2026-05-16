class TestLogger:
    """Plain-text logger for unit-test output."""

    _WIDTH = 72
    _LABEL_WIDTH = 18

    def print_suite_header(self, suite_name: str) -> None:
        line = "=" * self._WIDTH
        print(f"\n{line}")
        print(f"{suite_name.center(self._WIDTH)}")
        print(f"{line}\n")

    def print_group(self, group_name: str) -> None:
        print(f"\n{group_name}")
        print("-" * min(len(group_name), self._WIDTH))

    def print_result(self, test_name: str, passed: bool, details: str = "") -> None:
        status = "PASSED" if passed else "FAILED"
        msg = f"  {'Result':<{self._LABEL_WIDTH}}: [{status}] {test_name}"

        if details:
            msg += f"  ({details})"

        print(msg)

    def print_field(self, label: str, value) -> None:
        print(f"  {label:<{self._LABEL_WIDTH}}: {value}")

    def print_value(self, label: str, actual, expected=None) -> None:
        msg = f"  {label:<{self._LABEL_WIDTH}}: {actual}"

        if expected is not None:
            msg += f"  (expected: {expected})"

        print(msg)

    def print_warning(self, message: str, detail: str = "") -> None:
        msg = f"  Warning          : {message}"

        if detail:
            msg += f"  {detail}"

        print(msg)

    def print_info(self, message: str) -> None:
        print(f"  {message}")

    def print_summary(self, passed_count: int, total_count: int) -> None:
        failed_count = total_count - passed_count
        percent = passed_count / total_count * 100 if total_count > 0 else 0.0
        line = "=" * self._WIDTH

        print(f"\n{line}")
        print("  Summary")
        print(f"  {'Passed':<{self._LABEL_WIDTH}}: {passed_count}")
        print(f"  {'Failed':<{self._LABEL_WIDTH}}: {failed_count}")
        print(f"  {'Accuracy':<{self._LABEL_WIDTH}}: {percent:.1f}%")
        print(f"  {'Total':<{self._LABEL_WIDTH}}: {passed_count}/{total_count} passed")
        print(f"{line}\n")
