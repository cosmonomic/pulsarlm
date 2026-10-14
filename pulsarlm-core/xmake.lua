set_project("pulsarlm-core")
set_xmakever("3.0.0")

add_rules("mode.release", "mode.debug")
set_defaultmode("debug")
set_policy("build.optimization.lto", true)
add_ldflags("-flto=auto")
add_shflags("-flto=auto")

option("torch_cmake_prefix", {default = "", description = "Torch's CMake prefix path, used to configure libtorch."})
option("prof", {default = false, description = "Compile the PULSARLM_PROF phase timers into the engine."})
option("tests", {default = true, description = "Build the Catch2 test binaries."})
option("cuda_arch", {default = "native", description = "Comma-separated GPU architectures the CUDA kernels are compiled for, such as sm_89,sm_120,compute_120."})
option("digest", {default = "sha256", values = {"sha256", "blake3"}, description = "Hash function behind page digests."})

if is_config("digest", "blake3") then
    add_requires("blake3 1.8.7")
else
    add_requires("openssl3", {system = true})
end
add_requires("cmake::Torch", {system = true, configs = {
    envs = {CMAKE_PREFIX_PATH = get_config("torch_cmake_prefix")},
    presets = {CMAKE_LINK_DEPENDS_USE_LINKER = false},
}})

target("pulsarlm_core")
    set_kind("static")
    set_languages("c++23")
    add_options("cuda_arch")
    on_load(function (target)
        target:add("cugencodes", table.unpack(get_config("cuda_arch"):split(",")))
    end)
    set_values("cuda.rdc", false)
    add_cxflags("-fPIC")
    -- C++20, not the host code's C++23: nvcc 13.4's cicc crashes on any --c++23 input for aarch64.
    add_cuflags("-std=c++20", "-Xcompiler=-fPIC", "--expt-relaxed-constexpr", "--expt-extended-lambda")
    add_files("src/**.cu", "src/**.cpp|digest_*.cpp")
    add_files("src/digest_$(digest).cpp")
    add_includedirs("inc", {public = true})
    add_headerfiles("inc/(**)")
    add_packages("cmake::Torch", {public = true})
    if is_config("digest", "blake3") then
        add_packages("blake3")
    else
        add_packages("openssl3")
    end
    add_defines("PULSARLM_DIGEST_FUNCTION=\"$(digest)\"", {public = true})
    if has_config("prof") then
        add_defines("PULSARLM_PROF", {public = true})
    end
target_end()

if has_config("tests") then
    add_requires("catch2 3.5.4")

    for _, file in ipairs(os.files(path.join(os.scriptdir(), "tests/**/test_*.cpp"))) do
        target(path.basename(file))
            set_kind("binary")
            set_group("tests")
            set_languages("c++23")
            add_files(file)
            add_deps("pulsarlm_core")
            add_packages("catch2")
            add_tests("default")
        target_end()
    end
end
