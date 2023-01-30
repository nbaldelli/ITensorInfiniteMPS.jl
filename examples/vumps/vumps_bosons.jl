using ITensors, ITensors.HDF5
using Revise
using ITensorInfiniteMPS
using DelimitedFiles

include(
  joinpath(
    pkgdir(ITensorInfiniteMPS), "examples", "vumps", "src", "vumps_subspace_expansion.jl"
  ),
)

function ITensorInfiniteMPS.unit_cell_terms(::Model"QFM_model"; J1 = 1., J2 = 1., V = 1., U = 1.)
  #= hamiltonian definition =#
  ampo = OpSum()  
  ampo += -J1*(1), "Adag", 1, "A", 2
  ampo += -J1*(1), "Adag", 2, "A", 1

  ampo += -J1*(-1)^(1), "Adag", 2, "A", 3
  ampo += -J1*(-1)^(1), "Adag", 3, "A", 2

  ampo += V, "N", 1, "N", 2 #dens dens interaction
  ampo += V, "N", 2, "N", 3 #dens dens interaction

  ampo += -J2*exp(1im*10e-8), "Adag", 1, "A", 3
  ampo += -J2*exp(-1im*10e-8), "Adag", 3, "A", 1

  ampo += -J2*exp(1im*10e-8), "Adag", 2, "A", 4
  ampo += -J2*exp(-1im*10e-8), "Adag", 4, "A", 2

  ampo += U/2, "N", 1, "N", 1 #on-site interaction
  ampo += -U/2, "N", 1 
  ampo += U/2, "N", 2, "N", 2 #on-site interaction
  ampo += -U/2, "N", 2 
  return ampo
end

##############################################################################
#VUMPS parameters
#
function VUMPS(;
    J1 = 1., J2 = 0.4, V = 1., U = 6.,
    maxdim = 120, # Maximum bond dimension
    cutoff = 1e-7, # Singular value cutoff when increasing the bond dimension
    max_vumps_iters = 200, # Maximum number of iterations of the VUMPS algorithm at each bond dimension
    vumps_tol = 1e-5,
    outer_iters = 5, # Number of times to increase the bond dimension
    localham_type = MPO, # or ITensor
    conserve_qns = true,
    eager = true,
    reupload = false
    )
    ##############################################################################
    # CODE BELOW HERE DOES NOT NEED TO BE MODIFIED
    #

    N = 2 # Unit cell size

    @show N
    @show localham_type


    initstate(n) = isodd(n) ? "1" : "0" #half filling
    s = @show infsiteinds("Boson", N; dim = 4, initstate, conserve_qns)
    ψ = InfMPS(s, initstate)

    model = Model("QFM_model")
    model_params = (J1 = J1, J2 = J2, V = V, U = U)
    # Form the Hamiltonian 

    H = InfiniteSum{localham_type}(model, s; model_params...)
    #println(H)

    # Check translational invariance
    println("\nCheck translational invariance of initial infinite MPS")
    @show norm(contract(ψ.AL[1:N]..., ψ.C[N]) - contract(ψ.C[0], ψ.AR[1:N]...))

    outputlevel = 1
    vumps_kwargs = (tol=vumps_tol, maxiter=max_vumps_iters, outputlevel, eager)
    subspace_expansion_kwargs = (cutoff=cutoff, maxdim=maxdim)

    println("\nRun VUMPS on initial product state, unit cell size $N")
    ψ = vumps_subspace_expansion(H, ψ; outer_iters, subspace_expansion_kwargs, vumps_kwargs)

    h5open("MPS.h5","cw") do f
        if haskey(f, "psi_U($U)_V($V)_N($N)_J2($J2)_dim($maxdim)_V")
            delete_object(f, "psi_U($U)_V($V)_N($N)_J2($J2)_dim($maxdim)_V")
        end
        write(f,"psi_U($U)_V($V)_N($N)_J2($J2)_dim($maxdim)_V",ψ)
    end

    # Check translational invariance
    println("\nCheck translational invariance of optimized infinite MPS")
    @show norm(contract(ψ.AL[1:N]..., ψ.C[N]) - contract(ψ.C[0], ψ.AR[1:N]...))

    function expect_two_site(ψ::InfiniteCanonicalMPS, h::ITensor, n1n2)
        n1, n2 = n1n2
        ϕ = ψ.AL[n1] * ψ.AL[n2] * ψ.C[n2]
        return (noprime(ϕ * h) * dag(ϕ))[]
    end

    function correlation_matrix(ψ::InfiniteCanonicalMPS, op1, op2, dim)
        C = zeros(ComplexF64, dim, dim)
        for i in 1:dim
            for j in (i+1):dim
                h = op(op1, siteinds(ψ)[i]) * op(op2, siteinds(ψ)[j])
                ϕ = ψ.AL[i]
                for k in (i+1):(j)
                    ϕ *= ψ.AL[k]
                end
                ϕ *= ψ.C[j]
                C[i,j] = (noprime(ϕ * h) * dag(ϕ))[]
            end
        end
        return C + C'
    end

    function expect_three_site(ψ::InfiniteCanonicalMPS, h::MPO, n1n2n3)
        h = prod(h)
        n1, n2, n3 = n1n2n3
        ϕ = ψ.AL[n1] * ψ.AL[n2] * ψ.AL[n3] * ψ.C[n3]
        return (noprime(ϕ * h) * dag(ϕ))[]
    end

    function expect_two_site(ψ::InfiniteCanonicalMPS, h::MPO, n1n2)
        return expect_two_site(ψ, prod(h), n1n2)
    end

    function expect_two_site(ψ::MPS, h::ITensor, n1n2)
        n1, n2 = n1n2
        ψ = orthogonalize(ψ, n1)
        ϕ = ψ[n1] * ψ[n2]
        return (noprime(ϕ * h) * dag(ϕ))[]
    end

    function expect_three_site(ψ::MPS, h::ITensor, n1n2n3)
        n1, n2, n3 = n1n2n3
        ψ = orthogonalize(ψ, n1)
        ϕ = ψ[n1] * ψ[n2] * ψ[n3]
        return (noprime(ϕ * h) * dag(ϕ))[]
    end

    Nup = [expect(ψ, "N", n) for n in 1:N]
    dim = (0.5-Nup[1])-(0.5-Nup[2])
    C = correlation_matrix(ψ, "Adag", "A", 6)
    bow = zeros(ComplexF64, 4)
    for i in 1:4
        bow[i] = C[i,(i+1)]+C[(i+1),i]+C[(i+1),(i+2)]+C[(i+2),(i+1)]
    end

    bs = [(1,2,3), (2,3,4), (3,4,5)]
    energy_infinite = map(b -> expect_three_site(ψ, H[b[1]], b), bs)

    println("\nResults from VUMPS")
    @show V
    @show real.(energy_infinite)
    println("\nCDW order")
    @show dim
    println("\nBOW order")
    @show bow[1]
    return energy_infinite, real(dim), real(bow[1])
end

Vs = [0.7, 0.8, 0.85, 0.86, 0.87, 0.88, 0.945, 0.95, 0.955, 0.96, 0.97, 0.98, 0.99, 1.0, 1.1, 1.3]
dim = zeros(length(Vs))
bow = zeros(length(Vs))
J2 = 0.4; V = 1.; U = 6.
for (i,V) in enumerate(Vs)
    E, dim[i], bow[i] = VUMPS(; V = V, U = U, J2 = J2, reupload = false)
    open("iDMRG_data.txt", "a") do io
        writedlm(io, real.([U V J2 dim[i] abs(bow[i]) (E[1]+E[2])/2]))
    end
end

#why do i have to overload method and add the package name? 
#why can't i just use the function name?
#why should i include vumps_subspace_expansion directly and it is not in the package?