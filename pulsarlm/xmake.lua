includes(os.isdir(path.join(os.scriptdir(), "pulsarlm-core")) and "pulsarlm-core" or "../pulsarlm-core")

set_project("pulsarlm")
set_policy("build.optimization.lto", true)
add_ldflags("-flto=auto")
add_shflags("-flto=auto")

option("torch_cmake_prefix")
    after_check(function (option)
        if option:value() == "" then
            option:set_value(os.iorunv("python", {"-c", "import torch.utils; print(torch.utils.cmake_prefix_path)"}):trim())
        end
    end)
option_end()

option("python_include")
    set_default("")
    set_description("Python's C header directory, used to build the extension module.")
    after_check(function (option)
        if option:value() == "" then
            option:set_value(os.iorunv("python", {"-c", "import sysconfig; print(sysconfig.get_paths()['include'])"}):trim())
        end
    end)
option_end()

target("pulsarlm")
    set_kind("shared")
    set_languages("c++23")
    set_basename("pulsarlm_core")
    set_prefixname("")
    set_symbols("hidden")
    add_files("ext/*.cpp")
    add_deps("pulsarlm_core")
    add_packages("cmake::Torch")
    add_options("python_include")
    on_load(function (target)
        target:add("shflags", target:pkg("cmake::Torch"):get("ldflags"))
        target:add("sysincludedirs", get_config("python_include"))
        target:add("linkdirs", path.join(get_config("torch_cmake_prefix"), "..", "..", "lib"))
        target:add("links", "torch_python")
    end)
