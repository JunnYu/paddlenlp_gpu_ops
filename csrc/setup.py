import multiprocessing
import os
import subprocess
from site import getsitepackages

import paddle


paddle_includes = []
for site_packages_path in getsitepackages():
    paddle_includes.append(os.path.join(site_packages_path, "paddle", "include"))
    paddle_includes.append(os.path.join(site_packages_path, "paddle", "include", "third_party"))
    paddle_includes.append(os.path.join(site_packages_path, "nvidia", "cudnn", "include"))


def clone_git_repo(version, repo_url, destination_path):
    try:
        subprocess.run(["git", "clone", "-b", version, "--single-branch", repo_url, destination_path], check=True)
        return True
    except subprocess.CalledProcessError as e:
        print(f"Git clone {repo_url} operation failed with the following error: {e}")
        print("Please check your network connection or access rights to the repository.")
        print(
            "If the problem persists, please refer to the README file for instructions on how to manually download and install the necessary components."
        )
        return False


def get_gencode_flags(compiled_all=False):
    if not compiled_all:
        prop = paddle.device.cuda.get_device_properties()
        cc = prop.major * 10 + prop.minor
        return ["-gencode", "arch=compute_{0},code=sm_{0}".format(cc)]
    else:
        return [
            "-gencode",
            "arch=compute_80,code=sm_80",
            "-gencode",
            "arch=compute_75,code=sm_75",
            "-gencode",
            "arch=compute_70,code=sm_70",
        ]


def get_sm_version():
    prop = paddle.device.cuda.get_device_properties()
    cc = prop.major * 10 + prop.minor
    return cc


def run_single(func):
    p = multiprocessing.Process(target=func)
    p.start()
    p.join()


def run_multi(func_list):
    processes = []
    for func in func_list:
        processes.append(multiprocessing.Process(target=func))
        processes.append(multiprocessing.Process(target=func))
        processes.append(multiprocessing.Process(target=func))

    for p in processes:
        p.start()

    for p in processes:
        p.join()


cc_flag = get_gencode_flags(compiled_all=False)
cc = get_sm_version()


def setup_fast_ln():
    from paddle.utils.cpp_extension import CUDAExtension, setup

    setup(
        name="fast_ln",
        ext_modules=CUDAExtension(
            include_dirs=paddle_includes,
            sources=[
                "fast_ln/ln_api.cpp",
                "fast_ln/ln_bwd_semi_cuda_kernel.cu",
                "fast_ln/ln_fwd_cuda_kernel.cu",
            ],
            extra_compile_args={
                "cxx": ["-O3"],
                "nvcc": [
                    "-O3",
                    "-U__CUDA_NO_HALF_OPERATORS__",
                    "-U__CUDA_NO_HALF_CONVERSIONS__",
                    "-U__CUDA_NO_BFLOAT16_OPERATORS__",
                    "-U__CUDA_NO_BFLOAT16_CONVERSIONS__",
                    "-U__CUDA_NO_BFLOAT162_OPERATORS__",
                    "-U__CUDA_NO_BFLOAT162_CONVERSIONS__",
                    "-I./apex/contrib/layer_norm/",
                    "--expt-relaxed-constexpr",
                    "--expt-extended-lambda",
                    "--use_fast_math",
                ]
                + cc_flag,
            },
        ),
    )


def setup_fused_ln():
    from paddle.utils.cpp_extension import CUDAExtension, setup

    setup(
        name="fused_ln",
        ext_modules=CUDAExtension(
            include_dirs=paddle_includes,
            sources=[
                "fused_ln/layer_norm_cuda.cu",
            ],
            extra_compile_args={
                "cxx": ["-O3"],
                "nvcc": [
                    "-O3",
                    "-U__CUDA_NO_HALF_OPERATORS__",
                    "-U__CUDA_NO_HALF_CONVERSIONS__",
                    "-U__CUDA_NO_BFLOAT16_OPERATORS__",
                    "-U__CUDA_NO_BFLOAT16_CONVERSIONS__",
                    "-U__CUDA_NO_BFLOAT162_OPERATORS__",
                    "-U__CUDA_NO_BFLOAT162_CONVERSIONS__",
                    "-I./apex/contrib/layer_norm/",
                    "--expt-relaxed-constexpr",
                    "--expt-extended-lambda",
                    "--use_fast_math",
                    "-maxrregcount=50",
                ]
                + cc_flag,
            },
        ),
    )


def setup_paddlenlp_ops():
    from paddle.utils.cpp_extension import CUDAExtension, setup

    sources = [
        "paddlenlp_ops/save_with_output.cc",
        "paddlenlp_ops/set_value_by_flags.cu",
        "paddlenlp_ops/token_penalty_multi_scores.cu",
        "paddlenlp_ops/token_penalty_multi_scores_v2.cu",
        "paddlenlp_ops/stop_generation_multi_ends.cu",
        "paddlenlp_ops/fused_get_rope.cu",
        "paddlenlp_ops/get_padding_offset.cu",
        "paddlenlp_ops/qkv_transpose_split.cu",
        "paddlenlp_ops/rebuild_padding.cu",
        "paddlenlp_ops/transpose_removing_padding.cu",
        "paddlenlp_ops/write_cache_kv.cu",
        "paddlenlp_ops/encode_rotary_qk.cu",
        "paddlenlp_ops/get_padding_offset_v2.cu",
        "paddlenlp_ops/rebuild_padding_v2.cu",
        "paddlenlp_ops/set_value_by_flags_v2.cu",
        "paddlenlp_ops/stop_generation_multi_ends_v2.cu",
        "paddlenlp_ops/update_inputs.cu",
        "paddlenlp_ops/get_output.cc",
        "paddlenlp_ops/save_with_output_msg.cc",
        "paddlenlp_ops/write_int8_cache_kv.cu",
        "paddlenlp_ops/step.cu",
        "paddlenlp_ops/quant_int8.cu",
        "paddlenlp_ops/dequant_int8.cu",
        "paddlenlp_ops/flash_attn_bwd.cc",
        "paddlenlp_ops/tune_cublaslt_gemm.cu",
    ]
    if cc >= 80:
        sources += ["paddlenlp_ops/int8_gemm_with_cutlass/gemm_dequant.cu"]

    if cc >= 89:
        sources += [
            "paddlenlp_ops/fp8_gemm_with_cutlass/fp8_fp8_half_gemm.cu",
            "paddlenlp_ops/cutlass_kernels/fp8_gemm_fused/fp8_fp8_gemm_scale_bias_act.cu",
            "paddlenlp_ops/fp8_gemm_with_cutlass/fp8_fp8_fp8_dual_gemm.cu",
            "paddlenlp_ops/cutlass_kernels/fp8_gemm_fused/fp8_fp8_dual_gemm_scale_bias_act.cu",
        ]

    cutlass_dir = "paddlenlp_ops/cutlass_kernels/cutlass"
    if not os.path.exists(cutlass_dir) or not os.listdir(cutlass_dir):
        if not os.path.exists(cutlass_dir):
            os.makedirs(cutlass_dir)
        clone_git_repo("v3.5.0", "https://github.com/NVIDIA/cutlass.git", cutlass_dir)

    setup(
        name="paddlenlp_ops",
        ext_modules=CUDAExtension(
            include_dirs=paddle_includes,
            sources=sources,
            extra_compile_args={
                "cxx": ["-O3"],
                "nvcc": [
                    "-O3",
                    "-U__CUDA_NO_HALF_OPERATORS__",
                    "-U__CUDA_NO_HALF_CONVERSIONS__",
                    "-U__CUDA_NO_BFLOAT16_OPERATORS__",
                    "-U__CUDA_NO_BFLOAT16_CONVERSIONS__",
                    "-U__CUDA_NO_BFLOAT162_OPERATORS__",
                    "-U__CUDA_NO_BFLOAT162_CONVERSIONS__",
                    "-Ipaddlenlp_ops/cutlass_kernels",
                    "-Ipaddlenlp_ops/cutlass_kernels/cutlass/include",
                    "-Ipaddlenlp_ops/fp8_gemm_with_cutlass",
                    "-Ipaddlenlp_ops",
                ]
                + cc_flag,
            },
            libraries=["cublasLt"],
        ),
    )


def setup_causal_conv1d():
    from paddle.utils.cpp_extension import CUDAExtension, setup

    sources = [
        "causal_conv1d/causal_conv1d.cpp",
        "causal_conv1d/causal_conv1d_fwd.cu",
        "causal_conv1d/causal_conv1d_bwd.cu",
        "causal_conv1d/causal_conv1d_update.cu",
    ]

    if cc >= 75:
        cc_flag.append("-DCUDA_BFLOAT16_AVAILABLE")

    extra_compile_args = {
        "cxx": ["-O3"],
        "nvcc": [
            "-O3",
            "-U__CUDA_NO_HALF_OPERATORS__",
            "-U__CUDA_NO_HALF_CONVERSIONS__",
            "-U__CUDA_NO_BFLOAT16_OPERATORS__",
            "-U__CUDA_NO_BFLOAT16_CONVERSIONS__",
            "-U__CUDA_NO_BFLOAT162_OPERATORS__",
            "-U__CUDA_NO_BFLOAT162_CONVERSIONS__",
            "--expt-relaxed-constexpr",
            "--expt-extended-lambda",
            "--use_fast_math",
            "--ptxas-options=-v",
            "-lineinfo",
            "--threads",
            "4",
        ]
        + cc_flag,
    }

    setup(
        name="causal_conv1d_cuda_pd",
        ext_modules=CUDAExtension(
            sources=sources,
            extra_compile_args=extra_compile_args,
        ),
    )


def setup_selective_scan():
    from paddle.utils.cpp_extension import CUDAExtension, setup

    real_complex_list = ["real"]
    dtype_list = ["fp16", "fp32"]

    if cc > 75:
        dtype_list.insert(1, "bf16")
        cc_flag.append("-DCUDA_BFLOAT16_AVAILABLE")

    sources = [
        "selective_scan/selective_scan.cpp",
    ]
    for real_or_complex in real_complex_list:
        for dtype in dtype_list:
            sources.append(f"selective_scan/selective_scan_fwd_{dtype}_{real_or_complex}.cu")
            sources.append(f"selective_scan/selective_scan_bwd_{dtype}_{real_or_complex}.cu")

    extra_compile_args = {
        "cxx": ["-O3", "-std=c++17"],
        "nvcc": [
            "-O3",
            "-std=c++17",
            "-U__CUDA_NO_HALF_OPERATORS__",
            "-U__CUDA_NO_HALF_CONVERSIONS__",
            "-U__CUDA_NO_BFLOAT16_OPERATORS__",
            "-U__CUDA_NO_BFLOAT16_CONVERSIONS__",
            "-U__CUDA_NO_BFLOAT162_OPERATORS__",
            "-U__CUDA_NO_BFLOAT162_CONVERSIONS__",
            "--expt-relaxed-constexpr",
            "--expt-extended-lambda",
            "--use_fast_math",
            "--ptxas-options=-v",
            "-lineinfo",
            "--threads",
            "4",
        ]
        + cc_flag,
    }

    setup(
        name="selective_scan_cuda_pd",
        ext_modules=CUDAExtension(
            include_dirs=paddle_includes,
            sources=sources,
            extra_compile_args=extra_compile_args,
        ),
    )


if __name__ == "__main__":
    run_multi(
        [
            setup_fast_ln,
            setup_fused_ln,
            setup_paddlenlp_ops,
            setup_causal_conv1d,
            setup_selective_scan,
        ],
    )
