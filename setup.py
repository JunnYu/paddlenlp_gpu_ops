# Copyright 2024 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Simple check list from AllenNLP repo: https://github.com/allenai/allennlp/blob/main/setup.py

To create the package for PyPI.

1. Run `make pre-release` (or `make pre-patch` for a patch release) then run `make fix-copies` to fix the index of the
   documentation.

   If releasing on a special branch, copy the updated README.md on the main branch for the commit you will make
   for the post-release and run `make fix-copies` on the main branch as well.

2. Unpin specific versions from setup.py that use a git install.

3. Checkout the release branch (v<RELEASE>-release, for example v4.19-release), and commit these changes with the
   message: "Release: <RELEASE>" and push.

4. Manually trigger the "Nightly and release tests on main/release branch" workflow from the release branch. Wait for
   the tests to complete. We can safely ignore the known test failures.

5. Wait for the tests on main to be completed and be green (otherwise revert and fix bugs).

6. Add a tag in git to mark the release: "git tag v<RELEASE> -m 'Adds tag v<RELEASE> for PyPI'"
   Push the tag to git: git push --tags origin v<RELEASE>-release

7. Build both the sources and the wheel. Do not change anything in setup.py between
   creating the wheel and the source distribution (obviously).

   For the wheel, run: "python setup.py bdist_wheel" in the top level directory
   (This will build a wheel for the Python version you use to build it).

   For the sources, run: "python setup.py sdist"
   You should now have a /dist directory with both .whl and .tar.gz source versions.

   Long story cut short, you need to run both before you can upload the distribution to the
   test PyPI and the actual PyPI servers:

   python setup.py bdist_wheel && python setup.py sdist

8. Check that everything looks correct by uploading the package to the PyPI test server:

   twine upload dist/* -r pypitest
   (pypi suggests using twine as other methods upload files via plaintext.)
   You may have to specify the repository url, use the following command then:
   twine upload dist/* -r pypitest --repository-url=https://test.pypi.org/legacy/

   Check that you can install it in a virtualenv by running:
   pip install -i https://testpypi.python.org/pypi paddlenlp_gpu_ops

   If you are testing from a Colab Notebook, for instance, then do:
   pip install paddlenlp_gpu_ops && pip uninstall paddlenlp_gpu_ops
   pip install -i https://testpypi.python.org/pypi paddlenlp_gpu_ops

   Check you can run the following commands:
   python -c "from paddlenlp_gpu_ops import __version__; print(__version__)"

9. Upload the final version to the actual PyPI:
   twine upload dist/* -r pypi

10. Prepare the release notes and publish them on GitHub once everything is looking hunky-dory. You can use the following
    Space to fetch all the commits applicable for the release: https://huggingface.co/spaces/lysandre/github-release. Repo should
    be `TODO`. `tag` should be the previous release tag (v0.26.1, for example), and `branch` should be
    the latest release branch (v0.27.0-release, for example). It denotes all commits that have happened on branch
    v0.27.0-release after the tag v0.26.1 was created.

11. Run `make post-release` (or, for a patch release, `make post-patch`). If you were on a branch for the release,
    you need to go back to main before executing this.
"""

import os
import re
import shutil
import sys
import textwrap
from pathlib import Path

from setuptools import Command, find_packages, setup


# IMPORTANT:
# 1. all dependencies should be listed here with their version requirements if any
# 2. once modified, run: `make deps_table_update` to update src/paddlenlp_gpu_ops/dependency_versions_table.py
_deps = [
    "triton>=2.2.0",
    "paddlepaddle-gpu>=3.0.0",
    "paddlenlp>=3.0.0",
    "einops>=0.6.1",
    "isort>=5.5.4",
    "numpy",
    "parameterized",
    "protobuf>=3.20.3,<4",
    "pytest",
    "pytest-timeout",
    "pytest-xdist",
    "python>=3.8.0",
    "black",
    "hf-doc-builder>=0.3.0",
    "urllib3<=2.0.0",
    "ruff==0.1.5",
    "GitPython<3.1.19",
]

# this is a lookup table with items like:
#
# tokenizers: "huggingface-hub==0.8.0"
# packaging: "packaging"
#
# some of the values are versioned whereas others aren't.
deps = {b: a for a, b in (re.findall(r"^(([^!=<>~]+)(?:[!=<>~].*)?$)", x)[0] for x in _deps)}

# since we save this data in src/paddlenlp_gpu_ops/dependency_versions_table.py it can be easily accessed from
# anywhere. If you need to quickly access the data from this table in a shell, you can do so easily with:
#
# python -c 'import sys; from paddlenlp_gpu_ops.dependency_versions_table import deps; \
# print(" ".join([deps[x] for x in sys.argv[1:]]))' tokenizers datasets
#
# Just pass the desired package names to that script as it's shown with 2 packages above.
#
# If paddlenlp_gpu_ops is not yet installed and the work is done from the cloned repo remember to add `PYTHONPATH=src` to the script above
#
# You can then feed this for example to `pip`:
#
# pip install -U $(python -c 'import sys; from paddlenlp_gpu_ops.dependency_versions_table import deps; \
# print(" ".join([deps[x] for x in sys.argv[1:]]))' tokenizers datasets)
#


def deps_list(*pkgs):
    return [deps[pkg] for pkg in pkgs]


class DepsTableUpdateCommand(Command):
    """
    A custom command that updates the dependency table.
    usage: python setup.py deps_table_update
    """

    description = "build runtime dependency table"
    user_options = [
        # format: (long option, short option, description).
        (
            "dep-table-update",
            None,
            "updates src/paddlenlp_gpu_ops/dependency_versions_table.py",
        ),
    ]

    def initialize_options(self):
        pass

    def finalize_options(self):
        pass

    def run(self):
        entries = "\n".join([f'    "{k}": "{v}",' for k, v in deps.items()])
        content = [
            "# THIS FILE HAS BEEN AUTOGENERATED. To update:",
            "# 1. modify the `_deps` dict in setup.py",
            "# 2. run `make deps_table_update`",
            "deps = {",
            entries,
            "}",
            "",
        ]
        target = "src/paddlenlp_gpu_ops/dependency_versions_table.py"
        print(f"updating {target}")
        with open(target, "w", encoding="utf-8", newline="\n") as f:
            f.write("\n".join(content))


extras = {}
extras["quality"] = deps_list("urllib3", "isort", "ruff", "hf-doc-builder")
extras["docs"] = deps_list("hf-doc-builder")
extras["training"] = deps_list("paddlenlp", "protobuf", "triton", "einops")
extras["test"] = deps_list(
    "GitPython",
    "parameterized",
    "pytest",
    "pytest-timeout",
    "pytest-xdist",
)
extras["paddlepaddle-gpu"] = deps_list("paddlepaddle-gpu")

extras["dev"] = extras["quality"] + extras["test"] + extras["training"] + extras["docs"] + extras["paddlepaddle-gpu"]

install_requires = [
    deps["einops"],
    deps["triton"],
]

version_range_max = max(sys.version_info[1], 10) + 1


# NEW ADDED START
def write_custom_op_api_py(libname, filename):
    libname = str(libname)
    filename = str(filename)
    import paddle

    op_names = paddle.utils.cpp_extension.extension_utils.load_op_meta_info_and_register_op(libname)
    api_content = [paddle.utils.cpp_extension.extension_utils._custom_api_content(op_name) for op_name in op_names]
    dirname = os.path.dirname(os.path.abspath(filename))
    if not os.path.exists(dirname):
        os.makedirs(dirname)

    _stub_template = textwrap.dedent(
        """
        # THIS FILE IS GENERATED FROM PADDLEPADDLE SETUP.PY

        {custom_api}

        import os
        import sys
        import types
        import paddle
        import importlib.abc
        import importlib.util

        cur_dir = os.path.dirname(os.path.abspath(__file__))
        so_path = os.path.join(cur_dir, "lib/{resource}")

        def __bootstrap__():
            assert os.path.exists(so_path)
            # load custom op shared library with abs path
            custom_ops = paddle.utils.cpp_extension.load_op_meta_info_and_register_op(so_path)

            if os.name == 'nt' or sys.platform.startswith('darwin'):
                # Cpp Extension only support Linux now
                mod = types.ModuleType(__name__)
            else:
                try:
                    spec = importlib.util.spec_from_file_location(__name__, so_path)
                    assert spec is not None
                    mod = importlib.util.module_from_spec(spec)
                    assert isinstance(spec.loader, importlib.abc.Loader)
                    spec.loader.exec_module(mod)
                except ImportError:
                    mod = types.ModuleType(__name__)

            for custom_op in custom_ops:
                setattr(mod, custom_op, eval(custom_op))

        __bootstrap__()

        """
    ).lstrip()

    with open(filename, "w", encoding="utf-8") as f:
        f.write(_stub_template.format(resource=os.path.basename(libname), custom_api="\n\n".join(api_content)))


if len(sys.argv) > 0 and "deps_table_update" not in sys.argv:
    # generate lib files
    lib_path = Path("src/paddlenlp_gpu_ops/cuda/lib")
    if lib_path.exists():
        shutil.rmtree(lib_path)
    lib_path.mkdir(exist_ok=True)
    (lib_path / "__init__.py").touch(exist_ok=True)
    has_built = False
    for so_file in Path("csrc").glob("**/*.so"):
        so_filename = so_file.name
        # so file
        new_so_filename = so_filename.replace(".so", "_pd.so")
        new_so_file = lib_path / new_so_filename
        # py file
        py_filename = so_filename.replace(".so", ".py")
        new_py_file = lib_path.parent / py_filename
        shutil.copyfile(so_file, new_so_file)
        write_custom_op_api_py(new_so_file, new_py_file)
        has_built = True

    if not has_built:
        raise RuntimeError("No cuda lib found. Please build cuda lib first. See details in csrc/README.md")

# NEW ADDED END
setup(
    name="paddlenlp_gpu_ops",
    version="0.1.0.dev0",  # expected format is one of x.y.z.dev0, or x.y.z.rc1 or x.y.z (no to dashes, yes to dots)
    description="PaddleNLP GPU OPS cuda & triton.",
    long_description=open("README.md", "r", encoding="utf-8").read(),
    long_description_content_type="text/markdown",
    keywords="paddlenlp gpu ops contain cuda & triton",
    license="Apache 2.0 License",
    author="PaddlePaddle",
    author_email="paddlenlp@baidu.com",
    url="https://github.com/PaddlePaddle/paddlenlp/paddlenlp_gpu_ops",
    package_dir={"": "src"},
    packages=find_packages("src"),
    package_data={"paddlenlp_gpu_ops.cuda.lib": ["*.so", "*.dll", "*.dylib"]},
    include_package_data=True,
    python_requires=">=3.8.0",
    install_requires=list(install_requires),
    extras_require=extras,
    classifiers=[
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Programming Language :: Python :: 3",
    ]
    + [f"Programming Language :: Python :: 3.{i}" for i in range(8, version_range_max)],
    cmdclass={"deps_table_update": DepsTableUpdateCommand},
)
