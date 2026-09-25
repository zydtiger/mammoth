"""Root-bound preparation and retirement for ordered local publication."""

from __future__ import annotations

import os
from collections.abc import Callable, Iterator
from contextlib import contextmanager, suppress
from pathlib import Path
from typing import Union

from mammoth.core._filesystem import windows
from mammoth.core._filesystem.publication import (
    PreparedArtifact,
    _prepare_windows_artifact,
    directory_open_flags,
    prepare_artifact_in_directory,
    sync_directory_descriptor,
)


def resolve_confined_path(root: Path, path: Path, *, role: str) -> Path:
    """Resolve one target without following its final path component."""
    resolved = path.parent.resolve() / path.name
    if resolved == root or not resolved.is_relative_to(root):
        raise ValueError(f"checkpoint {role} target is outside checkpoint_root: {path}")
    return resolved


def retire_confined_path(
    root: Path,
    path: Path,
    *,
    root_identity: tuple[int, int],
) -> bool:
    """Unlink one exact retirement path through a no-follow root-relative traversal."""
    ensure_confined_path_unchanged(root, path, role="retirement")
    try:
        directory_descriptor = open_confined_parent(
            root,
            path,
            create=False,
            root_identity=root_identity,
        )
    except FileNotFoundError:
        return False
    try:
        try:
            os.unlink(path.name, dir_fd=directory_descriptor)
        except FileNotFoundError:
            return False
        sync_directory_descriptor(directory_descriptor)
        return True
    finally:
        with suppress(OSError):
            os.close(directory_descriptor)


def ensure_confined_path_unchanged(root: Path, path: Path, *, role: str) -> None:
    """Reject a normalized plan path whose parent now resolves elsewhere."""
    normalized = resolve_confined_path(root, path, role=role)
    if normalized != path:
        raise RuntimeError(
            f"checkpoint {role} parent changed before filesystem effect: {path.parent}"
        )


def open_confined_parent(
    root: Path,
    path: Path,
    *,
    create: bool,
    root_identity: tuple[int, int],
) -> int:
    """Open a target parent by walking from its checkpoint root without symlinks."""
    ensure_confined_path_unchanged(root, path, role="target")
    relative_parent = path.parent.relative_to(root)
    directory_descriptor = os.open(root, directory_open_flags())
    try:
        opened_root_stat = os.fstat(directory_descriptor)
        opened_root_identity = (opened_root_stat.st_dev, opened_root_stat.st_ino)
        if opened_root_identity != root_identity:
            raise RuntimeError(f"checkpoint root changed before filesystem effect: {root}")
        for component in relative_parent.parts:
            created = False
            if create:
                try:
                    os.mkdir(component, dir_fd=directory_descriptor)
                except FileExistsError:
                    pass
                else:
                    created = True
            child_descriptor = os.open(
                component,
                directory_open_flags(),
                dir_fd=directory_descriptor,
            )
            if created:
                try:
                    sync_directory_descriptor(directory_descriptor)
                except BaseException:
                    with suppress(OSError):
                        os.close(child_descriptor)
                    raise
            parent_descriptor = directory_descriptor
            directory_descriptor = child_descriptor
            try:
                os.close(parent_descriptor)
            except BaseException:
                with suppress(OSError):
                    os.close(parent_descriptor)
                raise
        return directory_descriptor
    except BaseException:
        with suppress(OSError):
            os.close(directory_descriptor)
        raise


def ensure_checkpoint_root(root: Path) -> tuple[int, int]:
    """Create a resolved checkpoint root and durably link each new directory."""
    missing_components: list[str] = []
    existing_parent = root
    while True:
        try:
            directory_descriptor = os.open(existing_parent, directory_open_flags())
        except FileNotFoundError:
            missing_components.append(existing_parent.name)
            existing_parent = existing_parent.parent
        else:
            break

    try:
        for component in reversed(missing_components):
            created = False
            try:
                os.mkdir(component, dir_fd=directory_descriptor)
            except FileExistsError:
                pass
            else:
                created = True
            child_descriptor = os.open(
                component,
                directory_open_flags(),
                dir_fd=directory_descriptor,
            )
            if created:
                try:
                    sync_directory_descriptor(directory_descriptor)
                except BaseException:
                    with suppress(OSError):
                        os.close(child_descriptor)
                    raise
            parent_descriptor = directory_descriptor
            directory_descriptor = child_descriptor
            try:
                os.close(parent_descriptor)
            except BaseException:
                with suppress(OSError):
                    os.close(parent_descriptor)
                raise
        root_stat = os.fstat(directory_descriptor)
        return root_stat.st_dev, root_stat.st_ino
    finally:
        with suppress(OSError):
            os.close(directory_descriptor)


def require_descriptor_relative_filesystem() -> None:
    """Reject platforms without the operations required for confined durability."""
    required = (os.mkdir, os.open, os.rename, os.stat, os.unlink)
    unsupported = [
        operation.__name__ for operation in required if operation not in os.supports_dir_fd
    ]
    if unsupported:
        names = ", ".join(sorted(unsupported))
        raise NotImplementedError(
            "ordered checkpoint publication requires POSIX descriptor-relative "
            f"filesystem operations; unavailable: {names}"
        )


def anchor_root(root: Path) -> tuple[int, int]:
    """Create a root and capture the identity required by later publication."""
    if os.name == "nt":
        with windows.pin_directory(root, create=True):
            info = root.stat()
            return info.st_dev, info.st_ino
    require_descriptor_relative_filesystem()
    return ensure_checkpoint_root(root)


class PublicationRoot:
    """Prepare and retire files relative to one previously anchored root."""

    def __init__(self, root: Path, identity: tuple[int, int]) -> None:
        self.path = root
        self.identity = identity

    def prepare(
        self,
        destination: Path,
        writer: Callable[[Path], object],
        *,
        mode: Union[int, None],
        preserve_permissions: bool,
        inspect_serialized: Callable[[int], object],
    ) -> PreparedArtifact:
        """Prepare one target while retaining the resources needed for commit."""
        if os.name == "nt":
            ensure_confined_path_unchanged(self.path, destination, role="target")
            return _prepare_windows_artifact(
                destination,
                writer,
                mode=mode,
                preserve_permissions=preserve_permissions,
                inspect_serialized=inspect_serialized,
            )
        descriptor = open_confined_parent(
            self.path, destination, create=True, root_identity=self.identity
        )
        return prepare_artifact_in_directory(
            destination,
            writer,
            directory_descriptor=descriptor,
            mode=mode,
            preserve_permissions=preserve_permissions,
            inspect_serialized=inspect_serialized,
        )

    def validate_target(self, destination: Path) -> None:
        """Refuse publication after a target parent has changed."""
        ensure_confined_path_unchanged(self.path, destination, role="publication")

    def retire(self, path: Path) -> bool:
        """Remove one exact prior target after successful publication."""
        if os.name != "nt":
            return retire_confined_path(self.path, path, root_identity=self.identity)
        ensure_confined_path_unchanged(self.path, path, role="retirement")
        try:
            with windows.pin_directory(path.parent):
                path.unlink()
        except FileNotFoundError:
            return False
        return True


@contextmanager
def publication_root(root: Path, identity: tuple[int, int]) -> Iterator[PublicationRoot]:
    """Hold platform root resources throughout one ordered publication."""
    if os.name != "nt":
        yield PublicationRoot(root, identity)
        return
    with windows.pin_directory(root):
        info = root.stat()
        if (info.st_dev, info.st_ino) != identity:
            raise RuntimeError(f"checkpoint root changed before filesystem effect: {root}")
        yield PublicationRoot(root, identity)
