"""Comprehensive caching utilities for improved performance."""

import hashlib
import logging
import pickle
import time
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


class SimulationCache:
    """Handle caching of simulation data for performance optimization."""

    def __init__(self, cache_dir: Path = Path("cache")):
        """Initialize cache manager."""
        self.cache_dir = cache_dir
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.cache_version = "1.0"

    def generate_cache_key(self, prefix: str, inputs: list[str]) -> str:
        """Generate cache key based on input files and parameters."""
        hasher = hashlib.md5()

        # Add cache version
        hasher.update(self.cache_version.encode())

        for input_item in inputs:
            if isinstance(input_item, str | Path):
                input_path = Path(input_item)
                if input_path.exists():
                    # Add file modification time and size
                    stat = input_path.stat()
                    hasher.update(
                        f"{input_path}:{stat.st_mtime}:{stat.st_size}".encode()
                    )
                else:
                    # For string parameters
                    hasher.update(str(input_item).encode())
            else:
                hasher.update(str(input_item).encode())

        return hasher.hexdigest()[:16]  # Use first 16 characters

    def get_cache_path(self, cache_type: str, cache_key: str) -> Path:
        """Get cache file path for given type and key."""
        return self.cache_dir / f"{cache_type}_{cache_key}.pkl"

    def is_cache_valid(self, cache_file: Path, source_files: list[Path]) -> bool:
        """Check if cached data is still valid."""
        try:
            if not cache_file.exists():
                return False

            cache_mtime = cache_file.stat().st_mtime

            # Check if any source file is newer than cache
            for source_file in source_files:
                if source_file.exists() and source_file.stat().st_mtime > cache_mtime:
                    return False

            return True
        except (OSError, AttributeError):
            return False

    def save_cache(self, cache_type: str, cache_key: str, data: dict[str, Any]) -> None:
        """Save data to cache."""
        try:
            cache_path = self.get_cache_path(cache_type, cache_key)

            # Add metadata
            cache_data = {
                **data,
                "_cache_metadata": {
                    "version": self.cache_version,
                    "created_at": time.time(),
                    "cache_type": cache_type,
                    "cache_key": cache_key,
                },
            }

            with cache_path.open("wb") as f:
                pickle.dump(cache_data, f, protocol=pickle.HIGHEST_PROTOCOL)

            logger.info(f"Saved {cache_type} cache to {cache_path}")

        except Exception as e:
            logger.warning(f"Failed to save {cache_type} cache: {e}")

    def load_cache(self, cache_type: str, cache_key: str) -> dict[str, Any] | None:
        """Load data from cache."""
        try:
            cache_path = self.get_cache_path(cache_type, cache_key)

            with cache_path.open("rb") as f:
                data = pickle.load(f)

            # Verify cache version
            metadata = data.get("_cache_metadata", {})
            if metadata.get("version") != self.cache_version:
                logger.warning(f"Cache version mismatch for {cache_type}, rebuilding")
                return None

            # Remove metadata before returning
            if "_cache_metadata" in data:
                del data["_cache_metadata"]

            logger.info(f"Loaded {cache_type} cache from {cache_path}")
            return data

        except Exception as e:
            logger.warning(f"Failed to load {cache_type} cache: {e}")
            return None

    def clear_cache(self, cache_type: str | None = None) -> None:
        """Clear cached data."""
        try:
            pattern = f"{cache_type}_*.pkl" if cache_type else "*.pkl"

            removed_count = 0
            for cache_file in self.cache_dir.glob(pattern):
                cache_file.unlink()
                removed_count += 1

            if removed_count > 0:
                logger.info(f"Cleared {removed_count} cache files")
            else:
                logger.info("No cache files to clear")

        except Exception as e:
            logger.warning(f"Failed to clear cache: {e}")

    def get_cache_info(self) -> dict[str, Any]:
        """Get information about cached data."""
        try:
            cache_files = list(self.cache_dir.glob("*.pkl"))
            total_size = sum(f.stat().st_size for f in cache_files)

            cache_types = {}
            for cache_file in cache_files:
                cache_type = cache_file.stem.split("_")[0]
                if cache_type not in cache_types:
                    cache_types[cache_type] = {"count": 0, "size": 0}
                cache_types[cache_type]["count"] += 1
                cache_types[cache_type]["size"] += cache_file.stat().st_size

            return {
                "total_files": len(cache_files),
                "total_size_mb": total_size / (1024 * 1024),
                "cache_types": cache_types,
                "cache_dir": str(self.cache_dir),
            }

        except Exception as e:
            logger.warning(f"Failed to get cache info: {e}")
            return {}
