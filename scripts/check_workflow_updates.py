"""Report outdated versions referenced in the GitHub Actions workflows.

Checks three things across ``.github/workflows/*.yml`` and
``.readthedocs.yaml``:

1. Every ``uses: owner/repo@ref`` action against the action's latest
   GitHub release.
2. The Poetry version pinned for ``snok/install-poetry`` against the
   latest Poetry release on PyPI.
3. Every Python version referenced against the Python release cycle
   from https://endoflife.date (flags end-of-life and newly released
   versions).

Only the Python standard library is used. Set the ``GITHUB_TOKEN``
environment variable to avoid GitHub's unauthenticated API rate limit
(60 requests per hour).

Usage (from the repository root)::

    python scripts/check_workflow_updates.py

The script exits with status 1 if any update is suggested, so it can
also be used as a CI check.

"""

__authors__ = "Derek Knowles"
__date__ = "03 Nov 2026"

import os
import re
import sys
import json
import glob
import datetime
import urllib.error
import urllib.request

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
WORKFLOW_GLOB = os.path.join(REPO_ROOT, ".github", "workflows", "*.yml")
READTHEDOCS_PATH = os.path.join(REPO_ROOT, ".readthedocs.yaml")

USES_RE = re.compile(r"uses:\s*([\w.-]+)/([\w.-]+)[\w./-]*@([^\s#]+)")
POETRY_RE = re.compile(r"uses:\s*snok/install-poetry@[^\n]*\n"
                       r"(?:[ \t]+[^\n]*\n)*?[ \t]+version:\s*([\d.]+)")
PYTHON_RE = re.compile(r"(?:python-version|python)\s*:\s*(\[[^\]]*\]|\S+)")
VERSION_RE = re.compile(r"3\.\d+")


def _get_json(url):
    """Fetch and decode a JSON document.

    Parameters
    ----------
    url : str
        URL of the JSON document.

    Returns
    -------
    data : dict or list or None
        Decoded JSON, or None if the request failed.

    """
    request = urllib.request.Request(url, headers={"User-Agent":
                                                   "gnss_lib_py"})
    token = os.environ.get("GITHUB_TOKEN")
    if token and "api.github.com" in url:
        request.add_header("Authorization", "Bearer " + token)
    try:
        with urllib.request.urlopen(request, timeout=20) as response:
            return json.loads(response.read().decode())
    except (urllib.error.URLError, TimeoutError, ValueError) as error:
        print("  ! could not fetch " + url + ": " + str(error))
        return None


def _version_tuple(ref):
    """Convert a tag such as ``v4.2.1`` into ``(4, 2, 1)``.

    Parameters
    ----------
    ref : str
        Git tag or branch name.

    Returns
    -------
    version : tuple or None
        Tuple of integers, or None if ``ref`` is not a version tag.

    """
    match = re.fullmatch(r"v?(\d+(?:\.\d+)*)", ref)
    if match is None:
        return None
    return tuple(int(part) for part in match.group(1).split("."))


def _read_files():
    """Read all workflow files and the Read the Docs configuration.

    Returns
    -------
    contents : dict
        Mapping of path relative to the repository root to file text.

    """
    paths = sorted(glob.glob(WORKFLOW_GLOB))
    if os.path.exists(READTHEDOCS_PATH):
        paths.append(READTHEDOCS_PATH)
    contents = {}
    for path in paths:
        with open(path, "r", encoding="utf-8") as file:
            contents[os.path.relpath(path, REPO_ROOT)] = file.read()
    return contents


def _action_status(ref, latest):
    """Determine whether an action tag has an update available.

    Parameters
    ----------
    ref : str
        Currently pinned action git reference.
    latest : str or None
        Latest release tag from GitHub.

    Returns
    -------
    status : str
        Human-readable update status description.
    is_outdated : bool
        True if a newer version of the action is available.

    """
    current_version = _version_tuple(ref)
    latest_version = _version_tuple(latest) if latest else None

    if current_version is None or latest_version is None:
        return ("skipped (ref '" + ref + "' is a branch, not a "
                + "version tag; latest release " + str(latest) + ")", False)
    if latest_version[:len(current_version)] > current_version:
        return ("UPDATE AVAILABLE -> " + latest, True)
    return ("up to date (latest " + str(latest) + ")", False)


def check_actions(contents):
    """Compare each action reference with its latest GitHub release.

    Parameters
    ----------
    contents : dict
        Mapping of file name to file text.

    Returns
    -------
    outdated : int
        Number of outdated action references.

    """
    print("\nGitHub Actions")
    print("--------------")
    actions = {}
    for fname, text in contents.items():
        for owner, repo, ref in USES_RE.findall(text):
            actions.setdefault((owner + "/" + repo, ref), set()).add(fname)

    latest_cache = {}
    outdated = 0
    for (action, ref), fnames in sorted(actions.items()):
        if action not in latest_cache:
            release = _get_json("https://api.github.com/repos/"
                                + action + "/releases/latest")
            latest_cache[action] = release.get("tag_name") \
                                   if isinstance(release, dict) else None
        status, is_outdated = _action_status(ref, latest_cache[action])
        if is_outdated:
            outdated += 1
        print("  " + action + "@" + ref + ": " + status)
        print("      used in: " + ", ".join(sorted(fnames)))
    return outdated


def check_poetry(contents):
    """Compare the Poetry version pinned in CI with the latest release.

    Parameters
    ----------
    contents : dict
        Mapping of file name to file text.

    Returns
    -------
    outdated : int
        Number of outdated Poetry pins.

    """
    print("\nPoetry version in CI")
    print("--------------------")
    pypi = _get_json("https://pypi.org/pypi/poetry/json")
    latest = pypi["info"]["version"] if pypi else None
    outdated = 0
    for fname, text in contents.items():
        for pinned in POETRY_RE.findall(text):
            if latest and _version_tuple(pinned) < _version_tuple(latest):
                status = "UPDATE AVAILABLE -> " + latest
                outdated += 1
            else:
                status = "up to date (latest " + str(latest) + ")"
            print("  " + fname + ": poetry " + pinned + ": " + status)
    return outdated


def check_python(contents):
    """Compare referenced Python versions with the Python release cycle.

    Parameters
    ----------
    contents : dict
        Mapping of file name to file text.

    Returns
    -------
    outdated : int
        Number of end-of-life or missing Python versions.

    """
    print("\nPython versions")
    print("---------------")
    cycles = _get_json("https://endoflife.date/api/python.json")
    if not cycles:
        return 0
    today = datetime.date.today().isoformat()
    released = {cycle["cycle"]: cycle for cycle in cycles}

    used = {}
    for fname, text in contents.items():
        for value in PYTHON_RE.findall(text):
            for version in VERSION_RE.findall(value):
                used.setdefault(version, set()).add(fname)

    outdated = 0
    for version in sorted(used, key=_version_tuple):
        eol = released.get(version, {}).get("eol")
        if isinstance(eol, str) and eol <= today:
            status = "END OF LIFE since " + eol + " -> consider dropping"
            outdated += 1
        else:
            status = "supported (end of life " + str(eol) + ")"
        print("  " + version + ": " + status)
        print("      used in: " + ", ".join(sorted(used[version])))

    newest = max((cycle["cycle"] for cycle in cycles
                  if cycle.get("releaseDate", "9999") <= today),
                 key=_version_tuple)
    if newest not in used:
        print("  " + newest + ": NEWLY RELEASED and not tested in any "
              "workflow -> consider adding")
        outdated += 1
    return outdated


def main():
    """Run all checks and return a process exit code.

    Returns
    -------
    exit_code : int
        0 if everything is up to date, 1 if any update is suggested.

    """
    contents = _read_files()
    total = check_actions(contents) \
          + check_poetry(contents) \
          + check_python(contents)
    print("\n" + str(total) + " suggested update(s).")
    return 1 if total else 0


if __name__ == "__main__":
    sys.exit(main())
