set_project("DiffCloth")
set_languages("c++17")

if is_mode("release") then
    set_optimize("fastest")
end

add_requires("eigen","nanogui","pybind11","openmp", "glad","glm")

add_cflags("-fdump-rtl-expand")

target("DiffCloth")
    set_kind("binary")
    add_files("src/code/**.cpp")
    add_files("src/code/**.c")
    add_headerfiles("src/code/**.h")
    add_includedirs("external/LBFGSpp/include")

    add_packages("eigen","nanogui","pybind11","openmp", "glad", "glm")

    add_defines("SOURCE_PATH=\"$(projectdir)\"")

target("diffcloth_py")    
    add_packages("eigen","nanogui","pybind11","openmp", "glad", "glm")
    set_kind("shared")
    add_files("src/code/python_interface.cpp")
    add_files("src/code/engine/**.cpp")
    add_files("src/code/optimization/**.cpp")
    add_files("src/code/simulation/**.cpp")
    add_files("src/code/supports/**.cpp")
    add_files("src/code/**.c")

    add_includedirs("external/LBFGSpp/include")

    add_defines("SOURCE_PATH=\"$(projectdir)\"")

    add_links("gomp")
    -- add_linkdirs("/usr/local/lib")

    set_filename("diffcloth_py.so")
    if is_plat("windows") then
        set_extension(".pyd")
    end

    after_build(
        function (target) 
            local targetfile = target:targetfile()
            os.cp(targetfile, os.projectdir() .. "/pylib/")
        end
    )
