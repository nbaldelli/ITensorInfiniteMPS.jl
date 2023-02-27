using ITensors
using ITensorInfiniteMPS
using ITensorCorrelators

include(
  joinpath(
    pkgdir(ITensorInfiniteMPS), "examples", "vumps", "src", "vumps_subspace_expansion.jl"
  ),
)

##########################################################
#Simulation of Ising 2D as a benchmark for the Infinite MPS 

function ITensorInfiniteMPS.unit_cell_terms(::Model"QFM_model"; J1 = 1., J2 = 1., V = 1., U = 1.)
    #= hamiltonian definition =#
    ampo = OpSum()  
    ampo += -J1*(1), "Adag", 1, "A", 2
    ampo += -J1*(1), "Adag", 2, "A", 1
  
    ampo += -J1*(-1)^(1), "Adag", 2, "A", 3
    ampo += -J1*(-1)^(1), "Adag", 3, "A", 2
  
    ampo += V, "N", 1, "N", 2 #dens dens interaction
    ampo += V, "N", 2, "N", 3 #dens dens interaction
  
    ampo += -J2, "Adag", 1, "A", 3
    ampo += -J2, "Adag", 3, "A", 1
  
    ampo += -J2, "Adag", 2, "A", 4
    ampo += -J2, "Adag", 4, "A", 2
  
    ampo += U/2, "N", 1, "N", 1 #on-site interaction
    ampo += -U/2, "N", 1 
    ampo += U/2, "N", 2, "N", 2 #on-site interaction
    ampo += -U/2, "N", 2 
    return ampo
end

maxdim = 20 # Maximum bond dimension
cutoff = 1e-6 # Singular value cutoff when increasing the bond dimension
max_vumps_iters = 10 # Maximum number of iterations of the VUMPS/TDVP algorithm at a fixed bond dimension
tol = 1e-5 # Precision error tolerance for outer loop of VUMPS or TDVP
outer_iters = 5 # Number of times to increase the bond dimension
time_step = -Inf # -Inf corresponds to VUMPS, finite time_step corresponds to TDVP
solver_tol = (x -> x / 100) # Tolerance for the local solver (eigsolve in VUMPS and exponentiate in TDVP)
multisite_update_alg = "parallel" # Choose between ["sequential", "parallel"]. Only parallel works with TDVP.
conserve_qns = true # Whether or not to conserve spin parity
nsite = 1 # Number of sites in the unit cell
localham_type = ITensor # Can choose `ITensor` or `MPO`

# Parameters of the transverse field Ising model
model_params = (J=1.0, h=0.9)

##############################################################################
# CODE BELOW HERE DOES NOT NEED TO BE MODIFIED
#

model = Model("ising")

initstate(n) = "↑"
s = infsiteinds("S=1/2", nsite; initstate, conserve_szparity=conserve_qns)
ψ = InfMPS(s, initstate)

# Form the Hamiltonian
H = InfiniteSum{localham_type}(model, s; model_params...)

# Check translational invariance
@show norm(contract(ψ.AL[1:nsite]..., ψ.C[nsite]) - contract(ψ.C[0], ψ.AR[1:nsite]...))

vumps_kwargs = (
  tol=tol,
  maxiter=max_vumps_iters,
  solver_tol=solver_tol,
  multisite_update_alg=multisite_update_alg,
)
subspace_expansion_kwargs = (cutoff=cutoff, maxdim=maxdim)

ψ = vumps_subspace_expansion(H, ψ; outer_iters, subspace_expansion_kwargs, vumps_kwargs)


##########################################################
#Cut of the Infinite MPS by using deltas and orthogonalize
#TODO: this does not work as the 

function toMPS_2(ψ::InfiniteCanonicalMPS, start, stop)
    N = stop - start + 1
    ψc = (ψ.AL)[start:stop]
    s_start = inds(ψc[1])[1]
    s_stop = inds(ψ.C[stop])[2]

    M = MPS(N)
    M[1] = dag(ITensor([1. fill(0,dim(s_start)-1)...],s_start)) * ψc[1]
    for i in 2:(N-1)
        M[i] = ψc[i]
    end
    display(ψc[N])
    display(ψ.C[stop])
    display(dag(delta(s_stop)))

    M[N] = ψc[N] * ψ.C[stop] * dag( ITensor([1. fill(0,dim(s_stop)-1)...],s_stop))
    orthogonalize!(M,2)
    normalize!(M)
    return M
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

function correlation_matrix(ψ::MPS, op1, op2, dim)
    C = zeros(ComplexF64, dim, dim)
    for i in 1:dim
        orthogonalize!(ψ, i)
        for j in (i+1):dim
            h = op(op1, siteinds(ψ)[i]) * op(op2, siteinds(ψ)[j])
            ϕ = ψ
            for k in (i+1):(j)
                ϕ *= ψ[k]
            end
            C[i,j] = (noprime(ϕ * h) * dag(ϕ))[]
        end
    end
    return C + C'
end


start = 1; stop = 6
psi = toMPS_2(ψ, start, stop)

op_inds = []
for i in start:stop
    for j in (i+1):stop
        push!(op_inds, (i,j))
    end
end

A = ITensors.correlation_matrix(psi, "Sx", "Sx") 
psi2 = toMPS_2(ψ, start, stop+1)

C = ITensors.correlation_matrix(psi2, "Sx", "Sx")

E = correlation_matrix(ψ, "Sx", "Sx", 3)
F = correlation_matrix(ψ, "Sx", "Sx", 4)


display(A)
display(C)
display(E)
display(F)

pp = ψ[2:40]
orthogonalize!(pp,1)
normalize!(pp)
norm(pp)

C = ITensorCorrelators.correlator(pp, ("Sx", "Sx"), op_inds)
C = correlation_matrix(pp, "Sx", "Sx", 3)