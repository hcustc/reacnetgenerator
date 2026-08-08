# SPDX-License-Identifier: LGPL-3.0-or-later
# cython: language_level=3
# cython: linetrace=True
"""Provide utils for ReacNetGenerator."""

import asyncio
import hashlib
import itertools
import operator
import os
import pickle
import shutil
import tempfile
import threading
from collections.abc import Callable, Generator, Iterable
from contextlib import ExitStack
from multiprocessing import Pool, Semaphore
from typing import (
    IO,
    TYPE_CHECKING,
    Any,
    AnyStr,
    BinaryIO,
    Generic,
    Optional,
    cast,
    overload,
)

import lz4.frame
import numpy as np
import requests
from requests.adapters import HTTPAdapter
from tqdm.auto import tqdm

from ._logging import logger

_FRAME_INDEX_PICKLE_OVERHEAD_BYTES = 256
_ORDERED_SPOOL_MEMORY_BYTES = 1024 * 1024

if TYPE_CHECKING:
    import multiprocessing.pool
    import multiprocessing.synchronize

    import reacnetgenerator


class WriteBuffer(Generic[AnyStr]):
    """Store a buffer for writing files.

    It is expensive to write to a file, so we need to make a buffer.

    Parameters
    ----------
    f: fileObject
        The file object to write.
    linenumber: int, default: 1200
        The number of contents to store in the buffer. The buffer will be flushed
        if it exceeds the set number.
    sep: str or bytes, default: None
        The separator for contents. If None (default), there will be no separator.
    byte_limit: int, optional, default: None
        Flush before buffered encoded payload plus separators would exceed this
        many bytes. A single indivisible item may exceed the limit but is flushed
        immediately instead of being retained with later items.
    """

    def __init__(
        self,
        f: IO[AnyStr],
        linenumber: int = 1200,
        sep: AnyStr | None = None,
        byte_limit: int | None = None,
    ) -> None:
        self.f = f
        if sep is not None:
            self.sep = sep
        elif f.mode == "w":
            self.sep = cast(AnyStr, "")
        elif f.mode == "wb":
            self.sep = cast(AnyStr, b"")
        else:
            raise RuntimeError("File mode should be w or wb!")
        self.linenumber = linenumber
        if byte_limit is not None:
            byte_limit = int(byte_limit)
            if byte_limit <= 0:
                raise ValueError("byte_limit must be a positive integer")
        self.byte_limit = byte_limit
        self.buff: list[AnyStr] = []
        self._buffer_bytes = 0
        self.maximum_buffer_bytes = 0
        self._separator_bytes = self._encoded_size(self.sep)
        self.name = self.f.name

    @staticmethod
    def _encoded_size(text: AnyStr) -> int:
        return len(text.encode("utf-8")) if isinstance(text, str) else len(text)

    def append(self, text: AnyStr) -> None:
        """Append a text.

        Parameters
        ----------
        text : str or bytes
            The text to be appended.
        """
        item_bytes = self._encoded_size(text) + self._separator_bytes
        if (
            self.byte_limit is not None
            and self.buff
            and self._buffer_bytes + item_bytes > self.byte_limit
        ):
            self.flush()
        self.buff.append(text)
        self._buffer_bytes += item_bytes
        self.maximum_buffer_bytes = max(
            self.maximum_buffer_bytes,
            self._buffer_bytes,
        )
        self.check()

    def extend(self, text: Iterable[AnyStr]) -> None:
        """Extend texts.

        Parameters
        ----------
        text : list of strs or bytes
            Texts to be extended.
        """
        if self.byte_limit is None:
            self.buff.extend(text)
            self.check()
            return
        for item in text:
            self.append(item)

    def check(self) -> None:
        """Check if the number of stored contents exceeds.

        If so, the buffer will be flushed.
        """
        if len(self.buff) > self.linenumber or (
            self.byte_limit is not None and self._buffer_bytes >= self.byte_limit
        ):
            self.flush()

    def flush(self) -> None:
        """Flush the buffer."""
        if self.buff:
            self.f.writelines([cast(Any, self.sep).join(self.buff), self.sep])
            self.buff[:] = []
            self._buffer_bytes = 0

    def __enter__(self) -> "WriteBuffer[AnyStr]":
        """Enter the context."""
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        """Exit the context."""
        self.flush()
        self.f.__exit__(exc_type, exc_value, traceback)


@overload
def appendIfNotNone(f: WriteBuffer[str] | ExitStack, wbytes: str | None) -> None: ...
@overload
def appendIfNotNone(
    f: WriteBuffer[bytes] | ExitStack, wbytes: bytes | None
) -> None: ...
def appendIfNotNone(f: WriteBuffer[AnyStr] | ExitStack, wbytes: AnyStr | None) -> None:
    """Append a line to a file if the line is not None.

    Parameters
    ----------
    f : WriteBuffer
        The file to write.
    wbytes : str or bytes
        The line to write.
    """
    if wbytes is not None:
        assert not isinstance(f, ExitStack)
        f.append(wbytes)


def produce(
    semaphore: "multiprocessing.synchronize.Semaphore",
    plist: Iterable[Any],
    parameter: Any,
    cancel_event: threading.Event | None = None,
) -> Generator[tuple[Any, Any], None, None]:
    """Item producer with a semaphore.

    Prevent large memory usage due to slow IO.

    Parameters
    ----------
    semaphore : multiprocessing.Semaphore
        The semaphore to acquire.
    plist : list of objects
        The list of items to be passed.
    parameter : object
        The parameter yielded with each item.
    cancel_event : threading.Event, optional
        Stop producing while waiting for capacity when the consumer aborts.

    Yields
    ------
    item: object
        The item to be yielded.
    parameter: object
        The parameter yielded with each item.
    """
    for item in plist:
        if cancel_event is not None and cancel_event.is_set():
            return
        semaphore.acquire()
        if cancel_event is not None and cancel_event.is_set():
            return
        if parameter is not None:
            item = (item, parameter)
        yield item


def compress(x: str | bytes) -> bytes:
    """Compress the line.

    This function reduces IO overhead to speed up the program. The functions will
    use lz4 to compress, since lz4 has better performance
    that any others.

    The compressed format is size + data + size + data + ..., where size is a 64-bit
    little-endian integer.

    Parameters
    ----------
    x : str or bytes
        The line to compress.

    Returns
    -------
    bytes
        The compressed line, with a linebreak in the end.
    """
    if isinstance(x, str):
        x = x.encode()
    compress_block = lz4.frame.compress(x, compression_level=0)
    length_bytes = len(compress_block).to_bytes(64, byteorder="little")
    return length_bytes + compress_block


def decompress(x: bytes, isbytes: bool = False) -> str | bytes:
    """Decompress the line.

    Parameters
    ----------
    x : bytes
        The line to decompress.
    isbytes : bool, optional, default: False
        If the decompressed content is bytes. If not, the line will be decoded.

    Returns
    -------
    str or bytes
        The decompressed line.
    """
    x = lz4.frame.decompress(x[64:])
    if isbytes:
        return x
    return x.decode()


def listtobytes(x: Any) -> bytes:
    """Convert an object to a compressed line.

    Parameters
    ----------
    x : object
        The object to convert, such as numpy.ndarray.

    Returns
    -------
    bytes
        The compressed line.
    """
    return compress(pickle.dumps(x))


def _frame_indices_fit_signal_memory(frame_block: bytes, frame_count: int) -> bool:
    """Check whether decoded frame indices stay within one full bool signal."""
    frame_info = lz4.frame.get_frame_info(frame_block[64:])
    content_size = int(frame_info["content_size"])
    return (
        0
        < content_size
        <= (max(0, int(frame_count)) + _FRAME_INDEX_PICKLE_OVERHEAD_BYTES)
    )


def read_compressed_block(f: BinaryIO) -> Generator[bytes, None, None]:
    """Read compressed binary file, assuming the format is size + data + size + data + ...

    Parameters
    ----------
    f : fileObject
        The file object to read.

    Yields
    ------
    data: bytes
        The compressed block.
    """
    while True:
        sizeb = f.read(64)
        if not sizeb:
            break
        size = int.from_bytes(sizeb, byteorder="little")
        yield sizeb + f.read(size)


def _iter_compressed_record_fields(
    handle: BinaryIO,
    selected_fields: Iterable[int],
    *,
    field_count: int = 4,
) -> Generator[tuple[bytes, ...], None, None]:
    """Read selected fields without loading unselected compressed payloads."""
    field_count = int(field_count)
    selected_fields = tuple(int(field) for field in selected_fields)
    if field_count <= 0:
        raise ValueError("compressed record field count must be positive")
    if not selected_fields or len(set(selected_fields)) != len(selected_fields):
        raise ValueError("selected record fields must be unique and non-empty")
    if min(selected_fields) < 0 or max(selected_fields) >= field_count:
        raise ValueError("selected field is outside the compressed record")
    field_positions = [-1] * field_count
    for output_position, field_index in enumerate(selected_fields):
        field_positions[field_index] = output_position
    file_size = os.fstat(handle.fileno()).st_size
    position = handle.tell()
    while True:
        selected_blocks = [None] * len(selected_fields)
        for field_index, output_position in enumerate(field_positions):
            size_bytes = handle.read(64)
            position += len(size_bytes)
            if not size_bytes:
                if field_index == 0:
                    return
                raise EOFError("Incomplete compressed record")
            if len(size_bytes) != 64:
                raise EOFError("Truncated compressed block size")
            size = int.from_bytes(size_bytes, byteorder="little")
            if position + size > file_size:
                raise EOFError("Truncated compressed block payload")
            if output_position >= 0:
                selected_blocks[output_position] = size_bytes + handle.read(size)
            else:
                handle.seek(size, os.SEEK_CUR)
            position += size
        yield tuple(selected_blocks)


def bytestolist(x: bytes) -> Any:
    """Convert a compressed line to an object.

    Parameters
    ----------
    x : bytes
        The compressed line.

    Returns
    -------
    object
        The decompressed object.
    """
    data = decompress(x, isbytes=True)
    assert isinstance(data, bytes)
    return pickle.loads(data)


class _IndexedCallable:
    """Attach the input sequence index to an unordered worker result."""

    def __init__(self, func: Callable):
        self.func = func

    def __call__(self, indexed_item: tuple[int, Any]) -> tuple[int, Any]:
        index, item = indexed_item
        return index, self.func(item)


class _DiskOrderedResultSpool:
    """Keep encoded reorder data in bounded memory and spill excess to disk."""

    _GENERIC_RESULT = 0
    _STRING_RESULT = 1
    _BYTES_RESULT = 2
    _NONE_RESULT = 3
    _RAW_RESULT_BYTES = 4096
    _MEMORY_ENTRY_OVERHEAD_BYTES = 128

    def __init__(
        self,
        total: int,
        directory: str | None = None,
        *,
        memory_limit_bytes: int = 0,
    ):
        self.total = int(total)
        if self.total < 0:
            raise ValueError("ordered result total must be non-negative")
        self.memory_limit_bytes = int(memory_limit_bytes)
        if self.memory_limit_bytes < 0:
            raise ValueError("ordered result memory limit must be non-negative")
        self._temporary_directory = tempfile.TemporaryDirectory(
            prefix="reacnetgenerator-ordered-results-",
            dir=directory,
        )
        self.data_path = os.path.join(
            self._temporary_directory.name,
            "results.bin",
        )
        self.index_path = os.path.join(
            self._temporary_directory.name,
            "index.mmap",
        )
        self.data = None
        self.index = None
        self.write_offset = 0
        self.bytes_staged = 0
        self.bytes_written = 0
        self.max_file_bytes = 0
        self.memory: dict[int, bytes] = {}
        self.memory_bytes = 0
        self.max_memory_bytes = 0
        self.pending_count = 0
        self.max_pending = 0

    def __enter__(self) -> "_DiskOrderedResultSpool":
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.close()

    def _ensure_disk_storage(self) -> None:
        """Create the data file and sparse index only when a result spills."""
        if self.data is not None:
            return
        try:
            self.data = open(self.data_path, "w+b")
            # A newly truncated w+ mapping already reads as zero. Eagerly assigning
            # zero here would dirty every index page before any result arrives.
            self.index = np.memmap(
                self.index_path,
                mode="w+",
                dtype=np.uint64,
                shape=(max(1, self.total), 2),
            )
        except BaseException:
            self.close()
            raise

    def has(self, index: int) -> bool:
        return 0 <= index < self.total and (
            index in self.memory
            or (self.index is not None and bool(self.index[index, 1]))
        )

    @classmethod
    def _memory_cost(cls, block: bytes) -> int:
        """Conservatively account for one encoded dict entry and its payload."""
        return len(block) + cls._MEMORY_ENTRY_OVERHEAD_BYTES

    @classmethod
    def _encode_result(cls, value: Any) -> bytes:
        if isinstance(value, str):
            payload = value.encode(
                "utf-8",
                errors="surrogatepass",
            )
            if len(payload) <= cls._RAW_RESULT_BYTES:
                return bytes((cls._STRING_RESULT,)) + payload
        if isinstance(value, bytes):
            if len(value) <= cls._RAW_RESULT_BYTES:
                return bytes((cls._BYTES_RESULT,)) + value
        if value is None:
            return bytes((cls._NONE_RESULT,))
        return bytes((cls._GENERIC_RESULT,)) + listtobytes(value)

    @classmethod
    def _decode_result(cls, block: bytes) -> Any:
        if not block:
            raise RuntimeError("Ordered result spool contains an empty block")
        result_type = block[0]
        payload = block[1:]
        if result_type == cls._STRING_RESULT:
            return payload.decode("utf-8", errors="surrogatepass")
        if result_type == cls._BYTES_RESULT:
            return payload
        if result_type == cls._NONE_RESULT:
            if payload:
                raise RuntimeError(
                    "Ordered result spool contains an invalid None block"
                )
            return None
        if result_type == cls._GENERIC_RESULT:
            return bytestolist(payload)
        raise RuntimeError("Ordered result spool contains an unknown result type")

    def put(self, index: int, value: Any) -> None:
        if index < 0 or index >= self.total:
            raise RuntimeError("Ordered result index exceeds declared total")
        if self.has(index):
            raise RuntimeError("Ordered result contains a duplicate index")
        block = self._encode_result(value)
        self.bytes_staged += len(block)
        memory_cost = self._memory_cost(block)
        if self.memory_bytes + memory_cost <= self.memory_limit_bytes:
            self.memory[index] = block
            self.memory_bytes += memory_cost
            self.max_memory_bytes = max(self.max_memory_bytes, self.memory_bytes)
        else:
            self._ensure_disk_storage()
            self.data.seek(self.write_offset)
            self.data.write(block)
            self.index[index] = (self.write_offset, len(block))
            self.write_offset += len(block)
            self.bytes_written += len(block)
            self.max_file_bytes = max(self.max_file_bytes, self.write_offset)
        self.pending_count += 1
        self.max_pending = max(self.max_pending, self.pending_count)

    def pop(self, index: int) -> Any:
        if not self.has(index):
            raise RuntimeError("Ordered result is not available")
        block = self.memory.pop(index, None)
        if block is not None:
            self.memory_bytes -= self._memory_cost(block)
        else:
            assert self.data is not None
            assert self.index is not None
            offset, length = (int(value) for value in self.index[index])
            self.data.seek(offset)
            block = self.data.read(length)
            if len(block) != length:
                raise RuntimeError("Ordered result spool is truncated")
            self.index[index] = 0
        self.pending_count -= 1
        if self.pending_count == 0 and self.data is not None:
            self.data.seek(0)
            self.data.truncate(0)
            self.write_offset = 0
        return self._decode_result(block)

    def close(self) -> None:
        data = getattr(self, "data", None)
        if data is not None:
            data.close()
            self.data = None
        memory = getattr(self, "memory", None)
        if memory is not None:
            memory.clear()
            self.memory_bytes = 0
        index = getattr(self, "index", None)
        if index is not None:
            index.flush()
            mmap = getattr(index, "_mmap", None)
            if mmap is not None:
                mmap.close()
            self.index = None
        temporary_directory = getattr(self, "_temporary_directory", None)
        if temporary_directory is not None:
            temporary_directory.cleanup()
            self._temporary_directory = None


def listtostirng(
    l: str | list | tuple | np.ndarray, sep: list[str] | tuple[str, ...]
) -> str:
    """Convert a list to string, that is easier to store.

    Parameters
    ----------
    l : str or array-like
        The list to convert, which can contain any number of dimensions.
    sep : list of strs
        The seperators for each dimension.

    Returns
    -------
    str
        The converted string.
    """
    if isinstance(l, str):
        return l
    if isinstance(l, (list, tuple, np.ndarray)):
        return sep[0].join(listtostirng(x, sep[1:]) for x in l)
    return str(l)


def multiopen(
    pool: "multiprocessing.pool.Pool",
    func: Callable,
    l: IO,
    semaphore: Optional["multiprocessing.synchronize.Semaphore"] = None,
    nlines: int | None = None,
    unordered: bool = True,
    return_num: bool = False,
    start: int = 0,
    extra: Any | None = None,
    interval: int | None = None,
    bar: bool = True,
    desc: str | None = None,
    unit: str = "it",
    total: int | None = None,
    chunksize: int = 100,
    include_input_index: bool = False,
    cancel_event: threading.Event | None = None,
) -> Iterable:
    """Return an interated object for process a file with multiple processors.

    Parameters
    ----------
    pool : multiprocessing.Pool
        The pool for multiprocessing.
    func : function
        The function to process lines.
    l : File object
        The file object.
    semaphore : multiprocessing.Semaphore, optional, default: None
        The semaphore to acquire. If None (default), the object will be passed
        without control.
    nlines : int, optional, default: None
        The number of lines to pass to the function each time. If None (default),
        only one line will be passed to the function.
    unordered : bool, optional, default: True
        Whether the process can be unordered.
    return_num : bool, optional, default: False
        If True, adds a counter to an iterable.
    start : int, optional, default: 0
        The start number of the counter.
    extra : object, optional, default: None
        The extra object passed to the item.
    interval : int, optional, default: None
        The interval of items that will be passed to the function. For example,
        if set to 10, a item will be passed once every 10 items and others will
        be dropped.
    bar : bool, optional, default: True
        If True, show a tqdm bar for the iteration.
    desc : str, optional, default: None
        The description of the iteration shown in the bar.
    unit : str, optional, default: it
        The unit of the iteration shown in the bar.
    total : int, optional, default: None
        The total number of the iteration shown in the bar.
    include_input_index : bool, optional, default: False
        If True, run workers unordered and return ``(input_index, result)`` pairs.
        This is an internal mechanism used by :func:`run_mp` for disk-backed
        ordered output.
    cancel_event : threading.Event, optional
        Stop the semaphore-controlled producer if result consumption aborts.

    Returns
    -------
    object
        An object that can be iterated.
    """
    obj = l
    if nlines:
        obj = itertools.zip_longest(*[obj] * nlines)
    if interval:
        obj = itertools.islice(obj, 0, None, interval)
    if return_num:
        obj = enumerate(obj, start)
    if semaphore:
        obj = produce(semaphore, obj, extra, cancel_event)
    if include_input_index:
        obj = enumerate(obj)
        func = _IndexedCallable(func)
    chunksize = int(chunksize)
    if chunksize <= 0:
        raise ValueError("chunksize must be a positive integer")
    if unordered or include_input_index:
        obj = pool.imap_unordered(func, obj, chunksize)
    else:
        obj = pool.imap(func, obj, chunksize)
    if bar:
        obj = tqdm(obj, desc=desc, unit=unit, total=total, disable=None)
    return obj


class SCOUROPTIONS:
    """Scour (SVG optimization) options."""

    strip_xml_prolog = True
    remove_titles = True
    remove_descriptions = True
    remove_metadata = True
    remove_descriptive_elements = True
    strip_comments = True
    enable_viewboxing = True
    strip_xml_space_attribute = True
    strip_ids = True
    shorten_ids = True
    newlines = False


class SharedRNGData:
    """Share ReacNetGenerator data with a class of the submodule.

    Parameters
    ----------
    rng: reacnetgenerator.ReacNetGenerator
        The centered ReacNetGenerator class.
    usedRNGKeys: list of strs
        Keys that needs to pass from ReacNetGenerator class to the submodule.
    returnedRNGKeys: list of strs
        Keys that needs to pass from the submodule to ReacNetGenerator class.
    extraNoneKeys: list of strs, optional, default: None
        Set keys to None, which will be used in the submodule.
    """

    def __init__(
        self,
        rng: "reacnetgenerator.ReacNetGenerator",
        usedRNGKeys: list[str],
        returnedRNGKeys: list[str],
        extraNoneKeys: list[str] | None = None,
    ) -> None:
        self.rng = rng
        self.returnedRNGKeys = returnedRNGKeys
        for key in usedRNGKeys:
            setattr(self, key, getattr(self.rng, key))
        for key in returnedRNGKeys:
            setattr(self, key, None)
        if extraNoneKeys is not None:
            for key in extraNoneKeys:
                setattr(self, key, None)

    def returnkeys(self) -> None:
        """Return back keys to ReacNetGenerator class."""
        for key in self.returnedRNGKeys:
            setattr(self.rng, key, getattr(self, key))


def checksha256(filename: str, sha256_check: str | list[str]):
    """Check sha256 of a file is correct.

    Parameters
    ----------
    filename : str
        The filename.
    sha256_check : str or list of strs
        The sha256 to be checked.

    Returns
    -------
    bool
        Indicate whether sha256 is correct.
    """
    if not os.path.isfile(filename):
        return
    h = hashlib.sha256()
    b = bytearray(128 * 1024)
    mv = memoryview(b)
    with open(filename, "rb", buffering=0) as f:
        for n in iter(lambda: f.readinto(mv), 0):
            h.update(mv[:n])
    sha256 = h.hexdigest()
    logger.info(f"SHA256 of {filename}: {sha256}")
    if sha256 in must_be_list(sha256_check):
        return True
    logger.warning("SHA256 is not correct.")
    logger.warning(open(filename).read())
    return False


async def download_file(
    urls: str | list[str], pathfilename: str, sha256: str | None
) -> str:
    """Download files from remote urls if not exists.

    Parameters
    ----------
    urls: str or list of strs
        The url(s) that is available to download.
    pathfilename: str
        The downloading path of the file.
    sha256: str
        Sha256 of the file. If not None and match the file, the download will be skiped.

    Returns
    -------
    pathfilename: str
        The downloading path of the file.
    """
    s = requests.Session()
    s.mount("http://", HTTPAdapter(max_retries=3))
    s.mount("https://", HTTPAdapter(max_retries=3))
    # download if not exists
    if os.path.isfile(pathfilename) and (
        sha256 is None or checksha256(pathfilename, sha256)
    ):
        return pathfilename

    # from https://stackoverflow.com/questions/16694907
    for url in must_be_list(urls):
        logger.info(f"Try to download {pathfilename} from {url}")
        with s.get(url, stream=True) as r, open(pathfilename, "wb") as f:
            try:
                shutil.copyfileobj(r.raw, f)
                break
            except requests.exceptions.RequestException as e:
                logger.warning(f"Request {pathfilename} Error.", exc_info=e)
    else:
        raise RuntimeError(f"Cannot download {pathfilename}.")

    return pathfilename


async def gather_download_files(urls: list[dict]) -> None:
    """Asynchronously download files from remote urls if not exists.

    See download_multifiles function for details.

    See Also
    --------
    download_multifiles
    """
    await asyncio.gather(
        *[
            download_file(jdata["url"], jdata["fn"], jdata.get("sha256", None))
            for jdata in urls
        ]
    )


def download_multifiles(urls: list[dict]) -> None:
    """Download multiple files from dicts.

    Parameters
    ----------
    urls : list of dicts
        The information of download files. Each dict should contain the following key:
            - url: str or list of strs
                The url(s) that is available to download.
            - pathfilename: str
                The downloading path of the file.
            - sha256: str, optional, default: None
                Sha256 of the file. If not None and match the file, the download will be skiped.
    """
    asyncio.run(gather_download_files(urls))


class _SerialPool:
    """Minimal Pool interface for the zero-IPC ``nproc=1`` path."""

    @staticmethod
    def imap(func, iterable, chunksize=1):
        return map(func, iterable)

    imap_unordered = imap


class _SerialSemaphore:
    """Preserve ``produce`` input shaping without cross-process capacity."""

    @staticmethod
    def acquire():
        return True


def run_mp(
    nproc: int,
    *,
    max_inflight: int | None = None,
    initializer: Callable | None = None,
    initargs: tuple[Any, ...] = (),
    maxtasksperchild: int | None = 1000,
    disk_ordered: bool = False,
    ordered_spool_dir: str | None = None,
    ordered_spool_memory_bytes: int = _ORDERED_SPOOL_MEMORY_BYTES,
    **kwargs: Any,
) -> Iterable[Any]:
    """Process a file with multiple processors.

    Parameters
    ----------
    nproc : int
        The number of processors to be used.
    max_inflight : int, optional
        The maximum number of submitted inputs that have not yet been consumed or
        safely spooled.
    initializer : callable, optional
        A callable run once when each worker starts.
    initargs : tuple, optional
        Positional arguments passed to ``initializer``.
    maxtasksperchild : int, optional
        The number of tasks a worker may process before it is replaced.
    disk_ordered : bool, optional, default: False
        Preserve input order while allowing workers to finish unordered. Completed
        results waiting behind a slow earlier input use a bounded encoded cache and
        spill excess to disk, so they do not retain unbounded memory or block later
        task submission. Requires ``unordered=False`` and an exact ``total``.
    ordered_spool_dir : str, optional
        Parent directory for the temporary ordered-result spool. The spool is
        removed when iteration finishes or fails.
    ordered_spool_memory_bytes : int, optional
        Maximum estimated bytes of encoded out-of-order results retained before
        spilling to disk. Defaults to 1 MiB and remains strictly bounded.
    **kwargs : dict, optional
        Other parameters can be found in the `multiopen` method.

    Yields
    ------
    object
        The yielded object from the `multiopen` method.

    See Also
    --------
    multiopen
    """
    nproc = int(nproc)
    if nproc <= 0:
        raise ValueError("nproc must be a positive integer")
    declared_total = kwargs.get("total")
    if declared_total is not None:
        try:
            task_count = operator.index(declared_total)
        except TypeError:
            pass
        else:
            if task_count >= 0:
                nproc = min(nproc, max(1, task_count))
    chunksize = int(kwargs.get("chunksize", 100))
    if chunksize <= 0:
        raise ValueError("chunksize must be a positive integer")
    ordered_total = None
    if disk_ordered:
        if kwargs.get("unordered", True) is not False:
            raise ValueError("disk_ordered requires unordered=False")
        if kwargs.get("total") is None:
            raise ValueError("disk_ordered requires an exact total")
        ordered_total = int(kwargs["total"])
        if ordered_total < 0:
            raise ValueError("disk_ordered total must be non-negative")
        if "include_input_index" in kwargs:
            raise TypeError("include_input_index is managed by disk_ordered")
        ordered_spool_memory_bytes = int(ordered_spool_memory_bytes)
        if ordered_spool_memory_bytes < 0:
            raise ValueError("ordered spool memory limit must be non-negative")
        kwargs["include_input_index"] = True
    if max_inflight is None:
        max_inflight = nproc * 150
    max_inflight = int(max_inflight)
    if max_inflight < chunksize:
        raise ValueError("max_inflight must be greater than or equal to chunksize")
    if nproc == 1 and initializer is None:
        try:
            results = multiopen(
                pool=_SerialPool(),
                semaphore=_SerialSemaphore(),
                **kwargs,
            )
            if not disk_ordered:
                yield from results
                return
            assert ordered_total is not None
            next_index = 0
            for result_index, item in results:
                if int(result_index) != next_index:
                    raise RuntimeError("Ordered result contains a duplicate index")
                yield item
                next_index += 1
            if next_index != ordered_total:
                raise RuntimeError("Ordered result count does not match declared total")
            return
        except GeneratorExit:
            raise
        except BaseException:
            logger.exception("run_mp failed")
            raise
    pool = Pool(
        nproc,
        initializer=initializer,
        initargs=initargs,
        maxtasksperchild=maxtasksperchild,
    )
    semaphore = Semaphore(max_inflight)
    producer_cancel = threading.Event()
    try:
        results = multiopen(
            pool=pool,
            semaphore=semaphore,
            cancel_event=producer_cancel,
            **kwargs,
        )
        if not disk_ordered:
            for item in results:
                yield item
                semaphore.release()
        else:
            assert ordered_total is not None
            next_index = 0
            with _DiskOrderedResultSpool(
                ordered_total,
                ordered_spool_dir,
                memory_limit_bytes=ordered_spool_memory_bytes,
            ) as spool:
                for result_index, item in results:
                    result_index = int(result_index)
                    if result_index < 0 or result_index >= ordered_total:
                        raise RuntimeError(
                            "Ordered result index exceeds declared total"
                        )
                    if result_index < next_index or spool.has(result_index):
                        raise RuntimeError("Ordered result contains a duplicate index")
                    if result_index == next_index:
                        semaphore.release()
                        yield item
                        next_index += 1
                        while spool.has(next_index):
                            yield spool.pop(next_index)
                            next_index += 1
                    else:
                        spool.put(result_index, item)
                        del item
                        semaphore.release()
                while spool.has(next_index):
                    yield spool.pop(next_index)
                    next_index += 1
                if next_index != ordered_total or spool.pending_count:
                    raise RuntimeError(
                        "Ordered result count does not match declared total"
                    )
                logger.info(
                    "Ordered multiprocessing spool: %.3f MiB encoded, "
                    "%.3f MiB disk written, %.3f MiB peak file, "
                    "%.3f MiB peak memory, "
                    "%d maximum pending results",
                    spool.bytes_staged / (1024 * 1024),
                    spool.bytes_written / (1024 * 1024),
                    spool.max_file_bytes / (1024 * 1024),
                    spool.max_memory_bytes / (1024 * 1024),
                    spool.max_pending,
                )
    except:
        logger.exception("run_mp failed")
        producer_cancel.set()
        semaphore.release()
        pool.terminate()
        raise
    else:
        pool.close()
    finally:
        pool.join()


def must_be_list(obj: Any | list[Any]) -> list[Any]:
    """Convert a object to a list if the object is not a list.

    Parameters
    ----------
    obj : Object
        The object to convert.

    Returns
    -------
    obj: list
        If the input object is not a list, returns a list that only contains that
        object. Otherwise, returns that object.
    """
    if isinstance(obj, list):
        return obj
    return [obj]


def get_timestep_value(timestep: Any) -> Any:
    """Normalize stored timestep metadata to the timestep value."""
    if isinstance(timestep, tuple):
        timestep = timestep[-1]
    if isinstance(timestep, np.generic):
        return timestep.item()
    return timestep


def check_zero_signal(signal: np.ndarray) -> bool:
    """Check if the given signal contains only zeros.

    Parameters
    ----------
    signal : 1D array of bool
        The signal to check. The dtype should be bool.

    Returns
    -------
    bool
        False if the signal contains only zeros, True otherwise.
    """
    # Benchmark
    # one_million_ones = np.ones(10**6, dtype=bool)
    # %timeit reacnetgenerator.utils_np.check_zero_signal(one_million_ones)
    # 808 ns ± 197 ns per loop (mean ± std. dev. of 7 runs, 1,000,000 loops each)
    # %timeit one_million_ones.any()
    # 17.9 µs ± 1.01 µs per loop (mean ± std. dev. of 7 runs, 100,000 loops each)
    # %timeit one_million_ones[one_million_ones.argmax()]
    # 561 ns ± 135 ns per loop (mean ± std. dev. of 7 runs, 1,000,000 loops each)
    #
    # any() doesn't have short-circuits, but argmax() does for bool.
    # See https://stackoverflow.com/a/45774536/9567349
    return signal[signal.argmax()].item()


def idx_to_signal(idx: np.ndarray, step: int):
    """Convert an index array to a signal array.

    Parameters
    ----------
    idx : array_like
        Index array.
    step : int
        Step size.

    Returns
    -------
    signal : ndarray
        Signal array in int8.
    """
    # Cython implementation is 1-2x faster than Python implementation.
    # However, with it, it's hard to use the limited Python API:
    # (1) https://github.com/cython/cython/issues/5697
    # (2) memory view limited API only available since python 3.11
    # when 1 is resolved, we may add back the Cython implementation
    # for Python 3.11+ only (build two wheels).

    signal = np.zeros(step, dtype=np.int8)
    signal[idx] = 1
    return signal.reshape((step, 1))
