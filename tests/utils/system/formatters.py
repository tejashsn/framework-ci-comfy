"""Format helpers for RASTRA payload construction (subset of fleet tests.utils.system)."""


def format_memory_size(size_gb: int) -> str:
    if size_gb == 0:
        return "0 GB"
    if size_gb >= 1024:
        return f"{size_gb / 1024:.1f} TB"
    if size_gb < 1:
        return f"{size_gb * 1024:.0f} MB"
    return f"{size_gb} GB"


def format_cache_size(size_kb: int) -> str:
    if size_kb == 0:
        return "0 KB"
    if size_kb >= 1024:
        size_mb = size_kb / 1024
        if size_mb >= 1024:
            return f"{size_mb / 1024:.1f} GB"
        return f"{size_mb:.1f} MB"
    return f"{size_kb} KB"


def format_clock_speed(speed_mhz: int) -> str:
    if speed_mhz == 0:
        return "0 MHz"
    if speed_mhz >= 1000:
        return f"{speed_mhz / 1000:.2f} GHz"
    return f"{speed_mhz} MHz"
