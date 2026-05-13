class TestLogger:

    # ANSI color codes
    _GREEN  = "\033[92m"
    _RED    = "\033[91m"
    _YELLOW = "\033[93m"
    _CYAN   = "\033[96m"
    _GRAY   = "\033[90m"
    _BOLD   = "\033[1m"
    _RESET  = "\033[0m"

    # In tiêu đề lớn - gọi một lần đầu mỗi hàm/nhóm test.
    def print_suite_header(cls, suite_name: str) -> None:
        line = "=" * 60
        print(f"\n{cls._BOLD}{cls._CYAN}{line}")
        print(f"  {suite_name}")
        print(f"{line}{cls._RESET}\n")

    # In tiêu đề nhóm nhỏ, gọi trước một nhóm test liên quan.
    def print_group(cls, group_name: str) -> None:
        print(f"\n{cls._BOLD}{cls._YELLOW}--- {group_name} ---{cls._RESET}")

    # In một dòng kết quả PASSED / FAILED.
    def print_result(cls, test_name: str, passed: bool, details: str = "") -> None:
        if passed:
            status = f"{cls._GREEN}PASSED{cls._RESET}"
        else:
            status = f"{cls._RED}FAILED{cls._RESET}"

        msg = f"  [{status}] {test_name}"

        if details:
            msg += f"  {cls._GRAY}({details}){cls._RESET}"

        print(msg)

    # In giá trị actual (và expected nếu có) - tiện debug.
    def print_value(cls, label: str, actual, expected=None) -> None:
        print(f"  {cls._GRAY}{label}:{cls._RESET}  {actual}", end="")

        if expected is not None:
            print(f"  {cls._GRAY}(expected: {expected}){cls._RESET}", end="")

        print()

    # In cảnh báo
    def print_warning(cls, message: str, detail: str = "") -> None:
        msg = f"  {cls._YELLOW}WARNING: {message}{cls._RESET}"

        if detail:
            msg += f"  {cls._GRAY}{detail}{cls._RESET}"

        print(msg)

    # In thông tin phụ - màu xám nhạt
    def print_info(cls, message: str) -> None:
        print(f"  {cls._GRAY}{message}{cls._RESET}")

    # In tổng kết cuối suite - gọi sau khi chạy hết test.
    def print_summary(cls, passed_count: int, total_count: int) -> None:
        failed = total_count - passed_count
        line = "=" * 60

        print(f"\n{cls._BOLD}{line}")

        if failed == 0:
            color = cls._GREEN
        else:
            color = cls._RED

        print(
            f"  {color}{passed_count}/{total_count} passed"
            f"  |  {failed} failed{cls._RESET}"
        )

        print(f"{cls._BOLD}{line}{cls._RESET}\n")