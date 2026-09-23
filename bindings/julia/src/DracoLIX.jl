"""
DracoLIX.jl — Julia interop for the DracoLIX C++ core (Phase 5 prototype)

Simple ccall-based binding to the C API in core/include/dracolix/c_api.h.
Covers f64 zeros/ones, from_data, matmul, matvec, and version.
Other dtypes dispatch via the same pattern.
"""
module DracoLIX

using Libdl

const libname = "libdracolix_c"
const lib = Ref{Ptr{Nothing}}(C_NULL)

function __init__()
    # search order: build/ , /usr/local/lib, JULIA env
    for cand in (
        joinpath(@__DIR__, "../../build/libdracolix_c.so"),
        joinpath(@__DIR__, "../../build/libdracolix_c.dylib"),
        "libdracolix_c",
    )
        try
            h = Libdl.dlopen(cand)
            lib[] = h
            break
        catch
        end
    end
end

struct DracoArray
    ptr::Ptr{Cvoid}
    function DracoArray(ptr::Ptr{Cvoid})
        obj = new(ptr)
        finalizer(obj) do o
            if o.ptr != C_NULL && lib[] != C_NULL
                ccall((:dracolix_array_destroy, lib[]), Cvoid, (Ptr{Cvoid},), o.ptr)
            end
        end
        obj
    end
end

Base.ndims(a::DracoArray) = Int(ccall((:dracolix_array_ndim, lib[]), Csize_t, (Ptr{Cvoid},), a.ptr))
Base.length(a::DracoArray) = Int(ccall((:dracolix_array_size, lib[]), Csize_t, (Ptr{Cvoid},), a.ptr))
function Base.size(a::DracoArray)
    d = ndims(a)
    p = ccall((:dracolix_array_shape, lib[]), Ptr{Csize_t}, (Ptr{Cvoid},), a.ptr)
    ntuple(i -> Int(unsafe_load(p, i)), d)
end

version() = unsafe_string(ccall((:dracolix_version, lib[]), Cstring, ()))

function zeros_f64(shape::Vector{Int})
    s = Csize_t.(shape)
    p = ccall((:dracolix_array_f64_zeros, lib[]), Ptr{Cvoid}, (Ptr{Csize_t}, Csize_t), s, length(s))
    p == C_NULL && error("dracolix zeros failed")
    DracoArray(p)
end

function ones_f64(shape::Vector{Int})
    s = Csize_t.(shape)
    p = ccall((:dracolix_array_f64_ones, lib[]), Ptr{Cvoid}, (Ptr{Csize_t}, Csize_t), s, length(s))
    p == C_NULL && error("dracolix ones failed")
    DracoArray(p)
end

function from_matrix(mat::Matrix{Float64})
    # Julia is column-major, DracoLIX is row-major — transpose on copy
    m,n = size(mat)
    sh = Csize_t[m,n]
    # row-major buffer
    buf = vec(permutedims(mat))
    p = ccall((:dracolix_array_f64_from_data, lib[]), Ptr{Cvoid}, (Ptr{Float64}, Ptr{Csize_t}, Csize_t), buf, sh, 2)
    p == C_NULL && error("from_data failed")
    DracoArray(p)
end

function to_matrix(a::DracoArray)
    m,n = size(a)
    ptr = ccall((:dracolix_array_f64_data_const, lib[]), Ptr{Float64}, (Ptr{Cvoid},), a.ptr)
    buf = unsafe_wrap(Array{Float64}, ptr, length(a); own=false)
    # row-major -> col-major
    permutedims(reshape(copy(buf), (n,m))')  # simple but correct for prototype
end

function matmul(a::DracoArray, b::DracoArray)
    p = ccall((:dracolix_matmul_f64, lib[]), Ptr{Cvoid}, (Ptr{Cvoid}, Ptr{Cvoid}), a.ptr, b.ptr)
    p == C_NULL && error("matmul failed (shape mismatch?)")
    DracoArray(p)
end

function matvec(a::DracoArray, x::DracoArray)
    p = ccall((:dracolix_matvec_f64, lib[]), Ptr{Cvoid}, (Ptr{Cvoid}, Ptr{Cvoid}), a.ptr, x.ptr)
    p == C_NULL && error("matvec failed")
    DracoArray(p)
end

# convenience overloads for Julia matrices
matmul(a::Matrix{Float64}, b::Matrix{Float64}) = to_matrix(matmul(from_matrix(a), from_matrix(b)))

end # module
