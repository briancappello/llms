"""bin/build-engine: which source it builds, and --check-only.

Source selection runs for real against local git repositories; cmake and
ninja are stubs on PATH, so no compiler is involved. The stub install drops
a copy of `true` as llama-server, which passes the ldd isolation check.
"""

import os
from pathlib import Path
import shutil
import stat
import subprocess
import sys
import tempfile
import unittest

SCRIPT = Path(__file__).resolve().parents[1] / "bin" / "build-engine"

STUB_CMAKE = """#!/bin/sh
case "$1" in
  --build) exit 0 ;;
  --install)
    prefix=$(cat "$2/prefix"); mkdir -p "$prefix/bin"
    cp "@TRUE@" "$prefix/bin/llama-server"; exit 0 ;;
esac
prev=""; build=""; prefix=""
for a in "$@"; do
  [ "$prev" = "-B" ] && build="$a"
  case "$a" in -DCMAKE_INSTALL_PREFIX=*) prefix="${a#*=}" ;; esac
  prev="$a"
done
mkdir -p "$build" && echo "$prefix" > "$build/prefix"
"""

LINUX = sys.platform.startswith("linux")
TOOLS = all(shutil.which(t) for t in ("git", "ldd"))


def git(cwd, *args):
    return subprocess.run(["git", "-C", str(cwd), *args], check=True, capture_output=True, text=True).stdout.strip()


@unittest.skipUnless(LINUX and TOOLS, "Linux with git and ldd required")
class SourceTests(unittest.TestCase):
    def setUp(self):
        self.root = Path(tempfile.mkdtemp(prefix="build-engine-"))
        self.addCleanup(shutil.rmtree, self.root)
        stubs = self.root / "stubs"
        stubs.mkdir()
        # `command -v true` in sh names the builtin, not a file: embed the path.
        cmake = STUB_CMAKE.replace("@TRUE@", shutil.which("true"))
        for name, body in (("cmake", cmake), ("ninja", "#!/bin/sh\nexit 0\n")):
            path = stubs / name
            path.write_text(body)
            path.chmod(path.stat().st_mode | stat.S_IEXEC)
        self.env = {k: v for k, v in os.environ.items() if not k.startswith(("LLMS_", "GIT_"))}
        self.env.update(PATH=f"{stubs}:{os.environ['PATH']}", LLMS_OPT_DIR=str(self.root / "opt"),
                        GIT_AUTHOR_NAME="t", GIT_AUTHOR_EMAIL="t@t", GIT_COMMITTER_NAME="t",
                        GIT_COMMITTER_EMAIL="t@t", GIT_CONFIG_GLOBAL="/dev/null", GIT_CONFIG_NOSYSTEM="1")
        # origin (bare) <- writer pushes; src is the operator's --src checkout
        self.origin = self.root / "origin.git"
        subprocess.run(["git", "init", "--quiet", "--bare", "-b", "master", str(self.origin)], check=True, env=self.env)
        self.writer = self.root / "writer"
        subprocess.run(["git", "clone", "--quiet", str(self.origin), str(self.writer)], check=True,
                       env=self.env, capture_output=True)
        self.first = self.commit("v1")
        git(self.writer, "tag", "good")
        self.push()
        self.src = self.root / "src"
        subprocess.run(["git", "clone", "--quiet", str(self.origin), str(self.src)], check=True, env=self.env)

    def commit(self, text):
        (self.writer / "file.txt").write_text(text)
        subprocess.run(["git", "-C", str(self.writer), "add", "."], check=True, env=self.env)
        subprocess.run(["git", "-C", str(self.writer), "commit", "--quiet", "-m", text], check=True, env=self.env)
        return git(self.writer, "rev-parse", "HEAD")

    def push(self):
        subprocess.run(["git", "-C", str(self.writer), "push", "--quiet", "--tags", "origin", "master"],
                       check=True, env=self.env, capture_output=True)

    def build(self, *args):
        return subprocess.run([str(SCRIPT), "eng", "cpu", *args], env=self.env, capture_output=True,
                              text=True, timeout=60)

    def built_commit(self):
        return (self.root / "opt" / "llama.cpp-eng" / "COMMIT").read_text().split()[0]

    def test_src_is_fast_forwarded_to_latest(self):
        latest = self.commit("v2")
        self.push()
        result = self.build("--src", str(self.src))
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(git(self.src, "rev-parse", "HEAD"), latest)
        self.assertEqual(git(self.src, "symbolic-ref", "--short", "HEAD"), "master")  # still on the branch
        self.assertTrue(latest.startswith(self.built_commit()))

    def test_src_may_be_a_git_worktree(self):
        worktree = self.root / "wt"
        git(self.src, "worktree", "add", "--quiet", "--track", "-b", "wt", str(worktree), "origin/master")
        latest = self.commit("v2")
        self.push()
        result = self.build("--src", str(worktree))
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(git(worktree, "rev-parse", "HEAD"), latest)

    def test_ref_pins_a_fallback_sha_or_tag(self):
        self.commit("v2")
        self.push()
        for ref in (self.first, "good"):
            with self.subTest(ref=ref):
                result = self.build("--src", str(self.src), "--ref", ref)
                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertEqual(git(self.src, "rev-parse", "HEAD"), self.first)

    def test_detached_src_without_ref_moves_to_origin_default(self):
        git(self.src, "checkout", "--quiet", "--detach", self.first)
        latest = self.commit("v2")
        self.push()
        result = self.build("--src", str(self.src))
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(git(self.src, "rev-parse", "HEAD"), latest)

    def test_uncommitted_changes_stop_the_build_untouched(self):
        self.commit("v2")
        self.push()
        (self.src / "file.txt").write_text("local edit")
        result = self.build("--src", str(self.src))
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("uncommitted changes", result.stderr)
        self.assertEqual((self.src / "file.txt").read_text(), "local edit")
        self.assertEqual(git(self.src, "rev-parse", "HEAD"), self.first)

    def test_diverged_local_commit_is_never_discarded(self):
        self.commit("v2")
        self.push()
        (self.src / "mine.txt").write_text("mine")
        git(self.src, "add", ".")
        subprocess.run(["git", "-C", str(self.src), "commit", "--quiet", "-m", "mine"], check=True, env=self.env)
        mine = git(self.src, "rev-parse", "HEAD")
        result = self.build("--src", str(self.src))
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("cannot fast-forward", result.stderr)
        self.assertEqual(git(self.src, "rev-parse", "HEAD"), mine)

    def test_repo_with_src_is_refused(self):
        result = self.build("--src", str(self.src), "--repo", str(self.origin))
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("--repo applies to the managed clone", result.stderr)

    def test_managed_clone_takes_latest_or_a_tag(self):
        self.env["LLMS_SRC_DIR"] = str(self.root / "managed")
        repo = f"file://{self.origin}"
        latest = self.commit("v2")
        self.push()
        result = self.build("--repo", repo)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertTrue(latest.startswith(self.built_commit()))
        result = self.build("--repo", repo, "--ref", "good")  # tags were refs/heads-only before
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertTrue(self.first.startswith(self.built_commit()))


@unittest.skipUnless(LINUX and TOOLS, "Linux with git and ldd required")
class CheckOnlyTests(unittest.TestCase):
    def run_check(self, backend):
        with tempfile.TemporaryDirectory() as root:
            server = Path(root) / "llama.cpp-probe" / "bin" / "llama-server"
            server.parent.mkdir(parents=True)
            shutil.copy(shutil.which("true"), server)  # no libggml/libllama: isolated
            stubs = Path(root) / "stubs"
            stubs.mkdir()
            for name in ("cmake", "ninja"):
                (stubs / name).write_text("#!/bin/sh\nexit 0\n")
                (stubs / name).chmod(0o755)
            env = {k: v for k, v in os.environ.items() if k != "LLMS_GPU_ARCH"}
            env.update(LLMS_OPT_DIR=root, PATH=f"{stubs}:{os.environ['PATH']}")
            return subprocess.run([str(SCRIPT), "probe", backend, "--check-only"],
                                  env=env, capture_output=True, text=True, timeout=30)

    def test_gpu_backends_need_no_arch_or_toolchain(self):
        # A build needs --arch (cuda) or ROCm (hip); inspecting a prefix needs neither.
        for backend in ("cuda", "hip", "cpu"):
            with self.subTest(backend=backend):
                result = self.run_check(backend)
                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertIn("isolation OK", result.stdout)


if __name__ == "__main__":
    unittest.main()
