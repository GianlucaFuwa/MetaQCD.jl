var documenterSearchIndex = {"docs":
[{"location":"parameters/#Full-Parameter-list-(-default):","page":"Parameters","title":"Full Parameter list (= default):","text":"","category":"section"},{"location":"parameters/","page":"Parameters","title":"Parameters","text":"Base.@kwdef mutable struct PrintPhysicalParameters\n    L::NTuple{4, Int64} = (4, 4, 4, 4)\n    beta::Float64 = 5.7\n    NC::Int64 = 3\n    kind_of_gaction::String = \"wilson\"\n    numtherm::Int64 = 10\n    numsteps::Int64 = 100\n    inital::String = \"cold\"\n    update_method::Vector{String} = [\"HMC\"]\n    hb_maxit::Int64 = 10^5\n    numheatbath::Int64 = 4\n    metro_epsilon::Float64 = 0.1\n    metro_numhits::Int64 = 1\n    metro_target_acc::Float64 = 0.5\n    eo::Bool = true\n    or_algorithm::String = \"subgroups\"\n    numorelax::Int64 = 0\n    parity_update::Bool = false\nend\n\nBase.@kwdef mutable struct PrintBiasParameters\n    kind_of_bias::String = \"none\"\n    kind_of_cv::String = \"clover\"\n    numsmears_for_cv::Int64 = 4\n    rhostout_for_cv::Float64 = 0.125\n    is_static::Union{Bool, Vector{Bool}} = false\n    symmetric::Bool = false\n    stride::Int64 = 1\n    cvlims::NTuple{2, Float64} = (-7, 7)\n    biasfactor::Float64 = Inf\n    kinds_of_weights::Vector{String} = [\"tiwari\"]\n    usebiases::Vector{String} = [\"\"]\n    write_bias_every::Int64 = 1\n    # metadynamics specific\n    bin_width::Float64 = 1e-2\n    meta_weight::Float64 = 1e-3\n    penalty_weight::Float64 = 1000.0\n    # opes specific\n    barrier::Float64 = 0.0\n    sigma0::Float64 = 0.1\n    sigma_min::Float64 = 1e-6\n    fixed_sigma::Bool = false\n    no_Z::Bool = false\n    opes_epsilon::Float64 = 0.0\n    threshold::Float64 = 1.0\n    cutoff::Float64 = 0.0\n    # for parametric\n    bias_Q::Float64 = 0.0\n    bias_A::Float64 = 0.0\n    bias_Z::Float64 = 0.0\n    # tempering specific\n    tempering_enabled::Bool = false\n    numinstances::Int64 = 1\n    swap_every::Int64 = 1\n    non_metadynamics_updates::Int64 = 1\n    measure_on_all::Bool = false\nend\n\nBase.@kwdef mutable struct PrintSystemParameters\n    backend::String = \"cpu\"\n    float_type::String = \"float64\"\n    log_dir::String = \"\"\n    log_to_console::Bool = true\n    verboselevel::Int64 = 1\n    loadU_format::String = \"\"\n    loadU_dir::String = \"\"\n    loadU_fromfile::Bool = false\n    loadU_filename::String = \"\"\n    saveU_dir::String = \"\"\n    saveU_format::String = \"\"\n    saveU_every::Int64 = 1\n    randomseed::Union{UInt64, Vector{UInt64}} = 0x0000000000000000\n    measurement_dir::String = \"\"\n    bias_dir::Union{String, Vector{String}} = \"\"\n    overwrite::Bool = false\nend\n\nBase.@kwdef mutable struct PrintHMCParameters\n    hmc_trajectory::Float64 = 1\n    hmc_steps::Int64 = 10\n    hmc_friction::Float64 = π/2\n    hmc_integrator::String = \"Leapfrog\"\n    hmc_numsmear::Int64 = 0\n    hmc_rhostout::Float64 = 0.0\nend\n\nBase.@kwdef mutable struct PrintGradientFlowParameters\n    flow_integrator::String = \"euler\"\n    flow_num::Int64 = 1\n    flow_tf::Float64 = 0.1\n    flow_steps::Int64 = 10\n    flow_measure_every::Union{Int64, Vector{Int64}} = 1\nend\n\nBase.@kwdef mutable struct PrintMeasurementParameters\n    measurement_method::Vector{Dict} = Dict[]\nend","category":"page"},{"location":"fermion_actions/","page":"-","title":"-","text":"<!– # Fermion Actions –> <!––> <!– Instead of explicitly creating the Dirac operator, one can also create the corresponding –> <!– fermion action, with similar syntax: –> <!––> <!– @docs --> <!-- WilsonFermionAction --> <!-- –> <!––> <!– @docs --> <!-- StaggeredFermionAction --> <!-- –> <!––> <!– @docs --> <!-- StaggeredEOPreFermionAction --> <!-- –>","category":"page"},{"location":"biased_sampling/#Biased-Sampling-Methods","page":"Biased Sampling Methods","title":"Biased Sampling Methods","text":"","category":"section"},{"location":"biased_sampling/","page":"Biased Sampling Methods","title":"Biased Sampling Methods","text":"Modules = [MetaQCD.BiasModule]\nOrder = [:function, :type]","category":"page"},{"location":"biased_sampling/#MetaQCD.BiasModule.calc_weights-Tuple{Nothing, Vararg{Any}}","page":"Biased Sampling Methods","title":"MetaQCD.BiasModule.calc_weights","text":"Weighting schemes based on the ones compared in \nhttps://pubs.acs.org/doi/pdf/10.1021/acs.jctc.9b00867\n\n\n\n\n\n","category":"method"},{"location":"biased_sampling/#MetaQCD.BiasModule.Bias","page":"Biased Sampling Methods","title":"MetaQCD.BiasModule.Bias","text":"Bias(p::ParameterSet, U::Gaugefield; instance=1)\n\nContainer that holds general parameters of bias enhanced sampling, like the kind of CV, its smearing and filenames/-pointers relevant to the bias. Also holds the specific kind of bias (Metadynamics, OPES or Parametric for now). \nThe instance keyword is used in case of PT-MetaD and multiple walkers to assign the correct usebias to each stream.\n\n\n\n\n\n","category":"type"},{"location":"biased_sampling/#MetaQCD.BiasModule.Metadynamics","page":"Biased Sampling Methods","title":"MetaQCD.BiasModule.Metadynamics","text":"Metadynamics(; symmetric=true, stride=1, cvlims=(-6, 6), biasfactor=Inf,\n              bin_width=0.1, weight=0.01, penalty_weight=1000)\nMetadynamics(p::ParameterSet; instance=1)\n\nCreate an instance of a Metadynamics bias using the inputs or the parameters given in p.\n\nSpecifiable parameters\n\nsymmetric::Bool = true - If true, the bias is built symmetrically by updating for both cv and -cv at every update-iteration \nstride::Int64 = 1 - Number of iterations between updates; must be >0 \ncvlims::NTuple{2, Float64} = (-6, 6) - Minimum and maximum of the explorable cv-space; must be ordered \nbiasfactor::Float64 = Inf - Biasfactor for well-tempered Metadynamics; must be >1 \nbin_width::Float64 = 0.1 - Width of bins in histogram; must be >0 \nweight::Float64 = 0.01 - (Starting) Height of added Gaussians; must be positive \npenalty_weight::Float64 = 1000 - Penalty when cv is outside of cvlims; must be positive \n\n\n\n\n\n\n","category":"type"},{"location":"biased_sampling/#MetaQCD.BiasModule.OPES","page":"Biased Sampling Methods","title":"MetaQCD.BiasModule.OPES","text":"OPES(; symmetric=true, stride=1, cvlims=(-6, 6), barrier=30,\n     biasfactor=Inf, σ₀=0.1, σ_min=1e-6, fixed_σ=true, opes_epsilon=0.0,\n     no_Z=false, threshold=1.0, cutoff=0.0)\nOPES(p::ParameterSet; instance=1)\n\nCreate an instance of a OPES bias using the parameters given in p.\n\nSpecifiable parameters\n\nsymmetric::Bool = true - If true, the bias is built symmetrically by updating for both cv and -cv at every update-iteration \nstride::Int64 = 1 - Number of iterations between updates; must be >0 \ncvlims::NTuple{2, Float64} = (-6, 6) - Minimum and maximum of the explorable cv-space; must be ordered \nbarrier::Float64 = 30 - Estimate of height of action barriers \nbiasfactor::Float64 = Inf - Biasfactor for well-tempered OPES; must be >1 \nσ₀::Float64 = 0.1 - (Starting) width of kernels; must be >0 \nσ_min::Float64 = 1e-6 - Minimum width of kernels; must be >0 \nfixed_σ::Bool = true - If true, width if kernels decreases iteratively \nϵ::Float64 = exp(-barrier/(1-1/biasfactor)) - Determines maximum height of bias; must be >0 \nno_Z::Bool = false - If false normalization factor Z is dynamically adjusted \nthreshold::Float64 = 1.0 - Threshold distance for kernel merging; must be >0 \ncutoff::Float64 = sqrt(2barrier/(1-1/biasfactor)) - Cutoff value for kernels; must be >0 \npenalty::Float64 = exp(-0.5cutoff²) - Penalty for being outside kernel cutoff; must be >0\n\n\n\n\n\n","category":"type"},{"location":"biased_sampling/#MetaQCD.BiasModule.Parametric","page":"Biased Sampling Methods","title":"MetaQCD.BiasModule.Parametric","text":"Parametric(cvlims, penalty_weight, Q, A, Z)\nParametric(p::ParameterSet; instance=1)\n\nCreate an instance of a static Parametric bias using the inputs or the parameters given in p.\n\nSpecifiable parameters\n\ncvlims::NTuple{2, Float64} = (-6, 6) - Minimum and maximum of the explorable cv-space; must be ordered \npenalty_weight::Float64 = 1000 - Penalty when cv is outside of cvlims; must be positive \nQ::Float64 = 0 - Quadratic term in the bias \nA::Float64 = 0 - Amplitude of the cosine term in the bias \nZ::Float64 = 0 - Frequency of the cosine term in the bias \n\n\n\n\n\n\n","category":"type"},{"location":"dirac/#Dirac-Operators","page":"Dirac Operators","title":"Dirac Operators","text":"","category":"section"},{"location":"dirac/","page":"Dirac Operators","title":"Dirac Operators","text":"To create a Dirac operators, use the constructors below:","category":"page"},{"location":"dirac/","page":"Dirac Operators","title":"Dirac Operators","text":"WilsonDiracOperator","category":"page"},{"location":"dirac/#MetaQCD.DiracOperators.WilsonDiracOperator","page":"Dirac Operators","title":"MetaQCD.DiracOperators.WilsonDiracOperator","text":"WilsonDiracOperator(::Abstractfield, mass; anti_periodic=true, r=1, csw=0)\nWilsonDiracOperator(D::WilsonDiracOperator, U::Gaugefield)\n\nCreate a free Wilson Dirac Operator with mass mass and Wilson parameter r. If anti_periodic is true the fermion fields are anti periodic in the time direction. If csw ≠ 0, a clover term is included.  This object cannot be applied to a fermion vector, since it lacks a gauge background. A Wilson Dirac operator with gauge background is created by applying it to a Gaugefield U like D_gauge = D_free(U)\n\nType Parameters:\n\nB: Backend (CPU / CUDA / ROCm)\nT: Floating point precision\nTF: Type of the Fermionfield used to store intermediate results when using the        Hermitian version of the operator\nTG: Type of the underlying Gaugefield\nC: Boolean declaring whether the operator is clover improved or not\n\n\n\n\n\n","category":"type"},{"location":"dirac/","page":"Dirac Operators","title":"Dirac Operators","text":"StaggeredDiracOperator","category":"page"},{"location":"dirac/#MetaQCD.DiracOperators.StaggeredDiracOperator","page":"Dirac Operators","title":"MetaQCD.DiracOperators.StaggeredDiracOperator","text":"StaggeredDiracOperator(::Abstractfield, mass; anti_periodic=true)\nStaggeredDiracOperator(D::StaggeredDiracOperator, U::Gaugefield)\n\nCreate a free Staggered Dirac Operator with mass mass. If anti_periodic is true the fermion fields are anti periodic in the time direction. This object cannot be applied to a fermion vector, since it lacks a gauge background. A Staggered Dirac operator with gauge background is created by applying it to a Gaugefield U like D_gauge = D_free(U)\n\nType Parameters:\n\nB: Backend (CPU / CUDA / ROCm)\nT: Floating point precision\nTF: Type of the Fermionfield used to store intermediate results when using the        Hermitian version of the operator\nTG: Type of the underlying Gaugefield\n\n\n\n\n\n","category":"type"},{"location":"dirac/","page":"Dirac Operators","title":"Dirac Operators","text":"StaggeredEOPreDiracOperator","category":"page"},{"location":"dirac/#MetaQCD.DiracOperators.StaggeredEOPreDiracOperator","page":"Dirac Operators","title":"MetaQCD.DiracOperators.StaggeredEOPreDiracOperator","text":"StaggeredEOPreDiracOperator(::Abstractfield, mass; anti_periodic=true)\nStaggeredEOPreDiracOperator(\n    D::Union{StaggeredDiracOperator,StaggeredEOPreDiracOperator},\n    U::Gaugefield\n)\n\nCreate a free even-odd preconditioned Staggered Dirac Operator with mass mass. If anti_periodic is true the fermion fields are anti periodic in the time direction. This object cannot be applied to a fermion vector, since it lacks a gauge background. A Staggered Dirac operator with gauge background is created by applying it to a Gaugefield U like D_gauge = D_free(U)\n\nType Parameters:\n\nB: Backend (CPU / CUDA / ROCm)\nT: Floating point precision\nTF: Type of the Fermionfield used to store intermediate results when using the        Hermitian version of the operator\nTG: Type of the underlying Gaugefield\n\n\n\n\n\n","category":"type"},{"location":"viz/#Visualization:","page":"Visualization","title":"Visualization:","text":"","category":"section"},{"location":"viz/","page":"Visualization","title":"Visualization","text":"We include the ability to visualize your data. For that, you have pass the the directory under \"ensembles\" that contains your measurements, creating a MetaMeasuremnts object holding all the measurements in Dict where the keys are symbols denoting the observable.","category":"page"},{"location":"viz/","page":"Visualization","title":"Visualization","text":"ens = \"my_ensemble\"\nmeasurements = MetaMeasurements(ens)","category":"page"},{"location":"viz/","page":"Visualization","title":"Visualization","text":"Now we can plot a timeseries of any observables measured on the ensemble via the timeseries method:","category":"page"},{"location":"viz/","page":"Visualization","title":"Visualization","text":"timeseries(measurements, :myobservable)","category":"page"},{"location":"viz/","page":"Visualization","title":"Visualization","text":"For hadron correlators there is a special function hadroncorrelator that plots the mean values of all time slices (without statistical uncertainties). Just specify the hadron whose correlator you want to see:","category":"page"},{"location":"viz/","page":"Visualization","title":"Visualization","text":"hadroncorrelator(measurements, :pion; logscale=true, calc_meff=false, tf=0.0)","category":"page"},{"location":"viz/","page":"Visualization","title":"Visualization","text":"You can also create a holder of a bias potential and plot it. MetaQCD.jl creates the bias files with an extension that gives their type (.metad or .opes), but if you changed the extension you have to provide the bias type as a symbol under the kwarg which:","category":"page"},{"location":"viz/","page":"Visualization","title":"Visualization","text":"bias = MetaBias(myfile, which=:mytype)\nbiaspotential(bias)","category":"page"},{"location":"observables/","page":"-","title":"-","text":"<!– # Measure Observables –>","category":"page"},{"location":"utils/#Utility-Functions","page":"Utility Functions","title":"Utility Functions","text":"","category":"section"},{"location":"utils/","page":"Utility Functions","title":"Utility Functions","text":"Modules = [MetaQCD.Utils]\nOrder = [:function, :type]","category":"page"},{"location":"utils/#MetaQCD.Utils.cdot-Union{Tuple{N}, Tuple{T}, Tuple{StaticArraysCore.SArray{Tuple{N}, Complex{T}, 1, N}, StaticArraysCore.SArray{Tuple{N}, Complex{T}, 1, N}}} where {T, N}","page":"Utility Functions","title":"MetaQCD.Utils.cdot","text":"cdot(a, b)\n\nReturn the complex dot product of a and b\n\n\n\n\n\n","category":"method"},{"location":"utils/#MetaQCD.Utils.ckron-Union{Tuple{N}, Tuple{M}, Tuple{T}, Tuple{StaticArraysCore.SArray{Tuple{M}, Complex{T}, 1, M}, StaticArraysCore.SArray{Tuple{N}, Complex{T}, 1, N}}} where {T, M, N}","page":"Utility Functions","title":"MetaQCD.Utils.ckron","text":"ckron(a, b)\nckron(A, B)\n\nReturn the complex Kronecker(outer) product of vectors a and b, i.e. a ⊗ b†, or of two matrices A and B, i.e. A ⊗ B.\n\n\n\n\n\n","category":"method"},{"location":"utils/#MetaQCD.Utils.cmvmul-Union{Tuple{NM}, Tuple{M}, Tuple{N}, Tuple{T}, Tuple{StaticArraysCore.SArray{Tuple{M, N}, Complex{T}, 2, NM}, StaticArraysCore.SArray{Tuple{N}, Complex{T}, 1, N}}} where {T, N, M, NM}","page":"Utility Functions","title":"MetaQCD.Utils.cmvmul","text":"cmvmul(A, x)\n\nReturn the matrix-vector product of A and x\n\n\n\n\n\n","category":"method"},{"location":"utils/#MetaQCD.Utils.cmvmul_color-Union{Tuple{N2}, Tuple{M}, Tuple{N}, Tuple{T}, Tuple{StaticArraysCore.SArray{Tuple{N, N}, Complex{T}, 2, N2}, StaticArraysCore.SArray{Tuple{M}, Complex{T}, 1, M}}} where {T, N, M, N2}","page":"Utility Functions","title":"MetaQCD.Utils.cmvmul_color","text":"cmvmul_color(A, x)\n\nReturn the matrix-vector product of A and x, where A only acts on the color structure of x.\n\n\n\n\n\n","category":"method"},{"location":"utils/#MetaQCD.Utils.cmvmul_d-Union{Tuple{NM}, Tuple{M}, Tuple{N}, Tuple{T}, Tuple{StaticArraysCore.SArray{Tuple{M, N}, Complex{T}, 2, NM}, StaticArraysCore.SArray{Tuple{N}, Complex{T}, 1, N}}} where {T, N, M, NM}","page":"Utility Functions","title":"MetaQCD.Utils.cmvmul_d","text":"cmvmul_d(A, x)\n\nReturn the matrix-vector product of the adjoint of A and x\n\n\n\n\n\n","category":"method"},{"location":"utils/#MetaQCD.Utils.cmvmul_d_color-Union{Tuple{N2}, Tuple{M}, Tuple{N}, Tuple{T}, Tuple{StaticArraysCore.SArray{Tuple{N, N}, Complex{T}, 2, N2}, StaticArraysCore.SArray{Tuple{M}, Complex{T}, 1, M}}} where {T, N, M, N2}","page":"Utility Functions","title":"MetaQCD.Utils.cmvmul_d_color","text":"cmvmul_d_color(A, x)\n\nReturn the matrix-vector product of A† and x, where A† only acts on the colo structure of x.\n\n\n\n\n\n","category":"method"},{"location":"utils/#MetaQCD.Utils.cmvmul_spin_proj-Union{Tuple{is_adjoint}, Tuple{ρ}, Tuple{N2}, Tuple{M}, Tuple{N}, Tuple{T}, Tuple{StaticArraysCore.SArray{Tuple{N, N}, Complex{T}, 2, N2}, StaticArraysCore.SArray{Tuple{M}, Complex{T}, 1, M}, Val{ρ}}, Tuple{StaticArraysCore.SArray{Tuple{N, N}, Complex{T}, 2, N2}, StaticArraysCore.SArray{Tuple{M}, Complex{T}, 1, M}, Val{ρ}, Val{is_adjoint}}} where {T, N, M, N2, ρ, is_adjoint}","page":"Utility Functions","title":"MetaQCD.Utils.cmvmul_spin_proj","text":"cmvmul_spin_proj(A, x, ::Val{ρ}, ::Val{is_adjoint}=Val(false))\n\nReturn A * (1 ± γᵨ) * x where γᵨ is the ρ-th Euclidean gamma matrix in the Chiral basis. x is assumed to be a 4xN component complex vector. The third argument is ρ wrapped in a Val and must be within the range [-4,4]. Its sign determines the sign in front of the γᵨ matrix. If is_adjoint is true, A† is used instead of A.\n\n\n\n\n\n","category":"method"},{"location":"utils/#MetaQCD.Utils.cvmmul-Union{Tuple{NM}, Tuple{M}, Tuple{N}, Tuple{T}, Tuple{StaticArraysCore.SArray{Tuple{N}, Complex{T}, 1, N}, StaticArraysCore.SArray{Tuple{M, N}, Complex{T}, 2, NM}}} where {T, N, M, NM}","page":"Utility Functions","title":"MetaQCD.Utils.cvmmul","text":"cvmmul(x, A)\n\nReturn the vector-matrix product of x† and A.\n\n\n\n\n\n","category":"method"},{"location":"utils/#MetaQCD.Utils.cvmmul_color-Union{Tuple{N2}, Tuple{M}, Tuple{N}, Tuple{T}, Tuple{StaticArraysCore.SArray{Tuple{M}, Complex{T}, 1, M}, StaticArraysCore.SArray{Tuple{N, N}, Complex{T}, 2, N2}}} where {T, N, M, N2}","page":"Utility Functions","title":"MetaQCD.Utils.cvmmul_color","text":"cvmmul_color(x, A)\n\nReturn the matrix-vector product of x† and A, where A only acts on the color structure of x†.\n\n\n\n\n\n","category":"method"},{"location":"utils/#MetaQCD.Utils.cvmmul_d-Union{Tuple{NM}, Tuple{M}, Tuple{N}, Tuple{T}, Tuple{StaticArraysCore.SArray{Tuple{N}, Complex{T}, 1, N}, StaticArraysCore.SArray{Tuple{M, N}, Complex{T}, 2, NM}}} where {T, N, M, NM}","page":"Utility Functions","title":"MetaQCD.Utils.cvmmul_d","text":"cvmmul_d(x, A)\n\nReturn the vector-matrix product of x and the adjoint of A. x is implicitly assumed to be a column vector and therefore the adjoint of x is used\n\n\n\n\n\n","category":"method"},{"location":"utils/#MetaQCD.Utils.cvmmul_d_color-Union{Tuple{N2}, Tuple{M}, Tuple{N}, Tuple{T}, Tuple{StaticArraysCore.SArray{Tuple{M}, Complex{T}, 1, M}, StaticArraysCore.SArray{Tuple{N, N}, Complex{T}, 2, N2}}} where {T, N, M, N2}","page":"Utility Functions","title":"MetaQCD.Utils.cvmmul_d_color","text":"cvmmul_d_color(x, A)\n\nReturn the matrix-vector product of x† and A†, where A† only acts on the color structure of x†.\n\n\n\n\n\n","category":"method"},{"location":"utils/#MetaQCD.Utils.exp_iQ-Union{Tuple{StaticArraysCore.SArray{Tuple{3, 3}, Complex{T}, 2, 9}}, Tuple{T}} where T","page":"Utility Functions","title":"MetaQCD.Utils.exp_iQ","text":"exp_iQ(Q::SU{3,9,T}) where {T}\nexp_iQ(e::exp_iQ_su3{T}) where {T}\n\nCompute the exponential of a traceless Hermitian 3x3 matrix Q or return the exp_iQ field of the exp_iQ_su3{T}-object e. \nFrom Morningstar & Peardon (2008) arXiv:hep-lat/0311018v1\n\n\n\n\n\n","category":"method"},{"location":"utils/#MetaQCD.Utils.exp_iQ_coeffs-Union{Tuple{StaticArraysCore.SArray{Tuple{3, 3}, Complex{T}, 2, 9}}, Tuple{T}} where T","page":"Utility Functions","title":"MetaQCD.Utils.exp_iQ_coeffs","text":"exp_iQ_coeffs(Q::SU{3,9,T}) where {T}\n\nReturn a exp_iQ_su3 object that contains the exponential of Q and all parameters obtained in the Cayley-Hamilton algorithm that are needed for Stout force recursion.\n\n\n\n\n\n","category":"method"},{"location":"utils/#MetaQCD.Utils.gaussian_TA_mat-Union{Tuple{Type{T}}, Tuple{T}} where T","page":"Utility Functions","title":"MetaQCD.Utils.gaussian_TA_mat","text":"gaussian_TA_mat(::Type{T}) where {T}\n\nGenerate a normally distributed traceless anti-Hermitian 3x3 matrix with precision T.\n\n\n\n\n\n","category":"method"},{"location":"utils/#MetaQCD.Utils.gen_SU2_matrix-Union{Tuple{T}, Tuple{Any, Type{T}}} where T","page":"Utility Functions","title":"MetaQCD.Utils.gen_SU2_matrix","text":"gen_SU2_matrix(ϵ, ::Type{T}) where {T}\n\nGenerate a Matrix X ∈ SU(2) with precision T near the identity with spread ϵ. \nFrom Gattringer C. & Lang C.B. (Springer, Berlin Heidelberg 2010)\n\n\n\n\n\n","category":"method"},{"location":"utils/#MetaQCD.Utils.gen_SU3_matrix-Union{Tuple{T}, Tuple{Any, Type{T}}} where T","page":"Utility Functions","title":"MetaQCD.Utils.gen_SU3_matrix","text":"gen_SU3_matrix(ϵ, ::Type{T}) where {T}\n\nGenerate a Matrix X ∈ SU(3) with precision T near the identity with spread ϵ. \nFrom Gattringer C. & Lang C.B. (Springer, Berlin Heidelberg 2010)\n\n\n\n\n\n","category":"method"},{"location":"utils/#MetaQCD.Utils.kenney_laub-Union{Tuple{StaticArraysCore.SArray{Tuple{3, 3}, Complex{T}, 2, 9}}, Tuple{T}} where T","page":"Utility Functions","title":"MetaQCD.Utils.kenney_laub","text":"kenney_laub(M::SMatrix{3,3,Complex{T},9}) where {T}\n\nCompute the SU(3) matrix closest to M using the Kenney-Laub algorithm.\n\n\n\n\n\n","category":"method"},{"location":"utils/#MetaQCD.Utils.move-Tuple{CartesianIndex{4}, Any, Any, Any}","page":"Utility Functions","title":"MetaQCD.Utils.move","text":"move(s::SiteCoords, μ, steps, lim)\n\nMove a site s in the direction μ by steps steps with periodic boundary conditions. The maximum extent of the lattice in the direction μ is lim.\n\n\n\n\n\n","category":"method"},{"location":"utils/#MetaQCD.Utils.multr-Union{Tuple{T}, Tuple{N²}, Tuple{N}, Tuple{StaticArraysCore.SArray{Tuple{N, N}, Complex{T}, 2, N²}, StaticArraysCore.SArray{Tuple{N, N}, Complex{T}, 2, N²}}} where {N, N², T}","page":"Utility Functions","title":"MetaQCD.Utils.multr","text":"multr(A::SU{N,N²,T}, B::SU{N,N²,T}) where {N,N²,T}\n\nCalculate the trace of the product of two SU(N) matrices A and B of precision T.\n\n\n\n\n\n","category":"method"},{"location":"utils/#MetaQCD.Utils.rand_SU3-Union{Tuple{Type{T}}, Tuple{T}} where T","page":"Utility Functions","title":"MetaQCD.Utils.rand_SU3","text":"rand_SU3(::Type{T}) where {T}\n\nGenerate a random Matrix X ∈ SU(3) with precision T. \n\n\n\n\n\n\n","category":"method"},{"location":"utils/#MetaQCD.Utils.spin_proj-Union{Tuple{ρ}, Tuple{M}, Tuple{T}, Tuple{StaticArraysCore.SArray{Tuple{M}, Complex{T}, 1, M}, Val{ρ}}} where {T, M, ρ}","page":"Utility Functions","title":"MetaQCD.Utils.spin_proj","text":"spin_proj(x, ::Val{ρ})\n\nReturn (1 ± γᵨ) * x where γᵨ is the ρ-th Euclidean gamma matrix in the Chiral basis. and x is a 4xN component complex vector. The second argument is ρ wrapped in a Val and must be within the range [-4,4]. Its sign determines the sign in front of the γᵨ matrix.\n\n\n\n\n\n","category":"method"},{"location":"utils/#MetaQCD.Utils.spintrace-Union{Tuple{M}, Tuple{T}, Tuple{StaticArraysCore.SArray{Tuple{M}, Complex{T}, 1, M}, StaticArraysCore.SArray{Tuple{M}, Complex{T}, 1, M}}} where {T, M}","page":"Utility Functions","title":"MetaQCD.Utils.spintrace","text":"spintrace(a, b)\n\nReturn the complex Kronecker(outer) product of vectors a and b, summing over dirac indices, i.e. ∑ ᵨ aᵨ ⊗ bᵨ†\n\n\n\n\n\n","category":"method"},{"location":"utils/#MetaQCD.Utils.switch_sides-Union{Tuple{T}, Tuple{CartesianIndex{4}, Vararg{T, 5}}} where T<:Integer","page":"Utility Functions","title":"MetaQCD.Utils.switch_sides","text":"switch_sides(site::CartesianIndex, NX, NY, NZ, NT, NV)\n\nReturn the cartesian index equivalent to site but with opposite parity. E.g., switch_sides((1, 1, 1, 1), 4, 4, 4, 4, 256) = (1, 1, 1, 3) and reverse\n\n\n\n\n\n","category":"method"},{"location":"utils/#MetaQCD.Utils.σμν_spin_mul-Union{Tuple{ν}, Tuple{μ}, Tuple{M}, Tuple{T}, Tuple{StaticArraysCore.SArray{Tuple{M}, Complex{T}, 1, M}, Val{μ}, Val{ν}}} where {T, M, μ, ν}","page":"Utility Functions","title":"MetaQCD.Utils.σμν_spin_mul","text":"σμν_spin_mul(x, ::Val{μ}, ::Val{ν})\n\nReturn σμν * x where σμν = i/2 * [γμ, γν] with the gamma matrices in the Chiral basis and x is a 4xN component complex vector. The latter two arguments are μ and ν wrapped in a Val and must be within the range [1,4] with μ < ν\n\n\n\n\n\n","category":"method"},{"location":"gaugefields/#Creating-Gaugefields","page":"Creating Gaugefields","title":"Creating Gaugefields","text":"","category":"section"},{"location":"gaugefields/","page":"Creating Gaugefields","title":"Creating Gaugefields","text":"To create 4-dimensional SU(3) gauge field, use the constructor U = Gaugefield(...) (see below) and set the initial conditions with identity_gauges!(U) or random_gauges!(U).","category":"page"},{"location":"gaugefields/","page":"Creating Gaugefields","title":"Creating Gaugefields","text":"Gaugefield","category":"page"},{"location":"gaugefields/#MetaQCD.Gaugefields.Gaugefield","page":"Creating Gaugefields","title":"MetaQCD.Gaugefields.Gaugefield","text":"Gaugefield(NX, NY, NZ, NT, β; BACKEND=CPU, T=Float64, GA=WilsonGaugeAction)\nGaugefield(U::Gaugefield)\n\nCreates a Gaugefield on BACKEND, i.e. an array of link-variables (SU3 matrices with T precision) of size 4 × NX × NY × NZ × NT with coupling parameter β and gauge action GA or a zero-initialized copy of U\n\nSupported backends\n\nCPU \nCUDABackend \nROCBackend\n\nSupported gauge actions\n\nWilsonGaugeAction \nSymanzikTreeGaugeAction (Lüscher-Weisz) \nIwasakiGaugeAction \nDBW2GaugeAction\n\n\n\n\n\n","category":"type"},{"location":"gaugefields/","page":"Creating Gaugefields","title":"Creating Gaugefields","text":"Temporaryfield","category":"page"},{"location":"gaugefields/#MetaQCD.Gaugefields.Temporaryfield","page":"Creating Gaugefields","title":"MetaQCD.Gaugefields.Temporaryfield","text":"Temporaryfield(NX, NY, NZ, NT; backend=CPU(), T=Val(Float64))\nTemporaryfield(u::Abstractfield)\n\nCreates a Temporaryfield on backend, i.e. an array of 3-by-3 T-precision matrices of size 4 × NX × NY × NZ × NT or a zero-initialized Temporaryfield of the same size as u\n\nSupported backends\n\nCPU \nCUDABackend \nROCBackend\n\n\n\n\n\n","category":"type"},{"location":"gaugefields/","page":"Creating Gaugefields","title":"Creating Gaugefields","text":"CoeffField","category":"page"},{"location":"gaugefields/#MetaQCD.Gaugefields.CoeffField","page":"Creating Gaugefields","title":"MetaQCD.Gaugefields.CoeffField","text":"CoeffField(NX, NY, NZ, NT; backend=CPU(), T=Val(Float64))\nCoeffField(u::Abstractfield)\n\nCreates a CoeffField on backend, i.e. an array of T-precison exp_iQ_su3 objects of size 4 × NX × NY × NZ × NT or of the same size as u. The objects hold the Q-matrices and all the exponential parameters needed for stout-force recursion\n\nSupported backends\n\nCPU \nCUDABackend \nROCBackend\n\n\n\n\n\n","category":"type"},{"location":"updates/#Updating-a-Gaugefield","page":"Updating a Gaugefield","title":"Updating a Gaugefield","text":"","category":"section"},{"location":"updates/","page":"Updating a Gaugefield","title":"Updating a Gaugefield","text":"To update a Gaugefield simply use the update! function that takes 2 arguments.  The first is the actual update algorithm update_alg and the second is the Gaugefield U.","category":"page"},{"location":"updates/","page":"Updating a Gaugefield","title":"Updating a Gaugefield","text":"U = Gaugefield(...)\nrandom_gauges!(U)\n\nupdate!(update_alg, U) ","category":"page"},{"location":"updates/#Supported-Update-Algorithms","page":"Updating a Gaugefield","title":"Supported Update Algorithms","text":"","category":"section"},{"location":"updates/","page":"Updating a Gaugefield","title":"Updating a Gaugefield","text":"Metropolis","category":"page"},{"location":"updates/#MetaQCD.Updates.Metropolis","page":"Updating a Gaugefield","title":"MetaQCD.Updates.Metropolis","text":"Metropolis(U::Gaugefield{B,T,A,GA}, eo, ϵ, numhits, target_acc, or_alg, numOR) where {B,T,A,GA}\n\nCreate a Metropolis object.\n\nArguments\n\nU::Gaugefield{B,T,A,GA}: Gauge field object.\neo: Even-odd preconditioning.\nϵ: Step size for the update.\nnumhits: Number of Metropolis hits.\ntarget_acc: Target acceptance rate.\nor_alg: Overrelaxation algorithm.\nnumOR: Number of overrelaxation sweeps.\n\nReturns\n\nA Metropolis object with the specified parameters. The gauge action GA of the field U determines the iterator used. For the plaquette or Wilson action it uses a Checkerboard iterator and for rectangular actions it partitions the lattice into four sublattices.\n\n\n\n\n\n","category":"type"},{"location":"updates/","page":"Updating a Gaugefield","title":"Updating a Gaugefield","text":"Heatbath","category":"page"},{"location":"updates/#MetaQCD.Updates.Heatbath","page":"Updating a Gaugefield","title":"MetaQCD.Updates.Heatbath","text":"Heatbath(U::Gaugefield{B,T,A,GA}, MAXIT, numHB, or_alg, numOR) where {B,T,A,GA}\n\nCreate a Heatbath` object.\n\nArguments\n\nU: The gauge field on which the update is performed.\nMAXIT: The maximum iteration count in the Heatbath update.\nnumHB: The number of Heatbath sweeps.\nor_alg: The overrelaxation algorithm used.\nnumOR: The number of overrelaxation sweeps.\n\nReturns\n\nA Heatbath object with the specified parameters. The gauge action GA of the field U determines the iterator used. For the plaquette or Wilson action it uses a Checkerboard iterator and for rectangular actions it partitions the lattice into four sublattices.\n\n\n\n\n\n","category":"type"},{"location":"updates/","page":"Updating a Gaugefield","title":"Updating a Gaugefield","text":"Overrelaxation","category":"page"},{"location":"updates/#MetaQCD.Updates.Overrelaxation","page":"Updating a Gaugefield","title":"MetaQCD.Updates.Overrelaxation","text":"Overrelaxation(algorithm)\n\nCreate an Overrelaxation object, that can be used within a Metropolis or Heatbath update step.\n\nSupported Algorithms\n\n\"subgroups\": Cabibbo-Marinari SU(2) subgroup embedding scheme\n\"kenney_laub\": Kenney-Laub projection onto SU(3)\n\n\n\n\n\n","category":"type"},{"location":"updates/","page":"Updating a Gaugefield","title":"Updating a Gaugefield","text":"HMC","category":"page"},{"location":"updates/#MetaQCD.Updates.HMC","page":"Updating a Gaugefield","title":"MetaQCD.Updates.HMC","text":"HMC(\n    U,\n    integrator,\n    trajectory,\n    steps,\n    friction = 0,\n    numsmear = 0,\n    ρ_stout = 0;\n    hmc_logging = true,\n    fermion_action = nothing,\n    heavy_flavours = 0,\n    bias_enabled = false,\n    logdir = \"\",\n)\n\nCreate an HMC object, that can be used as an update algorithm.\n\nArguments\n\nU: The gauge field on which the update is performed.\nintegrator: The integrator used to evolve the field.\ntrajectory: The length of the HMC trajectory.\nsteps: The number of integrator steps within the trajectory.\nfriction: Friction factor in the GHMC algorithm. Has to be in the range [0, 1].\nnumsmear: Number of Stout smearing steps applied to the gauge action.\nρ_stout: Step length of the Stout smearing applied to the gauge action.\nhmc_logging: If true, creates a logfile in logdir containing information\n\non the trajectories, unless logdir = \"\"\n\nfermion_action: An AbstratFermionAction to initialize the appropriate fermion fields\nheavy_flavours: The number of non-degenerate heavy flavours, again to initialize the\n\nright number of fermion fields\n\nbias_enabled: If true, additional fields are initialized that are needed for Stout\n\nforce recursion when using a bias.\n\nSupported Integrators\n\nLeapfrog\nOMF2\nOMF2Slow\nOMF4\nOMF4Slow\n\nSupported Fermion Actions\n\nWilsonFermionAction\nStaggeredFermionAction\nStaggeredEOPreFermionAction\n\nReturns\n\nAn HMC object, which can be used as an argument in the update! function.\n\n\n\n\n\n","category":"type"},{"location":"usage/#Usage","page":"Usage","title":"Usage","text":"","category":"section"},{"location":"usage/","page":"Usage","title":"Usage","text":"If you just want to perform a simulation with some parameters, then","category":"page"},{"location":"usage/","page":"Usage","title":"Usage","text":"Set parameters using one of the templates in template folder\nFrom shell, do:","category":"page"},{"location":"usage/","page":"Usage","title":"Usage","text":"julia --threads=auto run.jl parameters.toml","category":"page"},{"location":"usage/","page":"Usage","title":"Usage","text":"or","category":"page"},{"location":"usage/","page":"Usage","title":"Usage","text":"Start Julia (with project):","category":"page"},{"location":"usage/","page":"Usage","title":"Usage","text":"julia --threads=auto --project","category":"page"},{"location":"usage/","page":"Usage","title":"Usage","text":"Import MetaQCD package (this may prompt you to install dependencies):","category":"page"},{"location":"usage/","page":"Usage","title":"Usage","text":"using MetaQCD","category":"page"},{"location":"usage/","page":"Usage","title":"Usage","text":"Begin Simulation with prepared parameter file \"parameters.toml\":","category":"page"},{"location":"usage/","page":"Usage","title":"Usage","text":"run_sim(\"parameters.toml\")","category":"page"},{"location":"#MetaQCD.jl","page":"MetaQCD.jl: Metadynamics in Lattice QCD","title":"MetaQCD.jl","text":"","category":"section"},{"location":"","page":"MetaQCD.jl: Metadynamics in Lattice QCD","title":"MetaQCD.jl: Metadynamics in Lattice QCD","text":"Inspired by the LatticeQCD.jl package by Akio Tomiya et al.","category":"page"},{"location":"#Features","page":"MetaQCD.jl: Metadynamics in Lattice QCD","title":"Features","text":"","category":"section"},{"location":"","page":"MetaQCD.jl: Metadynamics in Lattice QCD","title":"MetaQCD.jl: Metadynamics in Lattice QCD","text":"Simulations of 4D-SU(3) Yang-Mills (Pure Gauge) theory\nMetadynamics\nPT-MetaD\nSeveral update algorithms (HMC, Metropolis, Heatbath, Overrelaxation)\nSeveral symplectic integrators for HMC (Leapfrog, OMF2, OMF4)\nGradient flow with variable integrators (Euler, RK2, RK3, RK3W7)\nImproved Gauge actions (Symanzik tree, Iwasaki, DBW2)\nImproved Topological charge definitions (clover, rectangle clover-improved)\nSupport for CUDA and ROCm backends","category":"page"},{"location":"#Installation","page":"MetaQCD.jl: Metadynamics in Lattice QCD","title":"Installation","text":"","category":"section"},{"location":"","page":"MetaQCD.jl: Metadynamics in Lattice QCD","title":"MetaQCD.jl: Metadynamics in Lattice QCD","text":"First make sure you have a Julia version at or above 1.9.0 installed. You can use juliaup for that or just install the release from the Julia website.","category":"page"},{"location":"","page":"MetaQCD.jl: Metadynamics in Lattice QCD","title":"MetaQCD.jl: Metadynamics in Lattice QCD","text":"The package in not in the general registry. So you will have to either","category":"page"},{"location":"","page":"MetaQCD.jl: Metadynamics in Lattice QCD","title":"MetaQCD.jl: Metadynamics in Lattice QCD","text":"Add the package to your Julia environment via:","category":"page"},{"location":"","page":"MetaQCD.jl: Metadynamics in Lattice QCD","title":"MetaQCD.jl: Metadynamics in Lattice QCD","text":"julia> ] add https://github.com/GianlucaFuwa/MetaQCD.jl","category":"page"},{"location":"","page":"MetaQCD.jl: Metadynamics in Lattice QCD","title":"MetaQCD.jl: Metadynamics in Lattice QCD","text":"or","category":"page"},{"location":"","page":"MetaQCD.jl: Metadynamics in Lattice QCD","title":"MetaQCD.jl: Metadynamics in Lattice QCD","text":"Clone this repository onto your machine.\nOpen Julia in the directory which you cloned the repo into, with the project specific environment. This can either be done by starting Julia with the command line argument \"–project\" or by activating the environment within an opened Julia instance via the package manager:","category":"page"},{"location":"","page":"MetaQCD.jl: Metadynamics in Lattice QCD","title":"MetaQCD.jl: Metadynamics in Lattice QCD","text":"using Pkg\nPkg.activate(\".\")","category":"page"},{"location":"","page":"MetaQCD.jl: Metadynamics in Lattice QCD","title":"MetaQCD.jl: Metadynamics in Lattice QCD","text":"Or you can switch to package manager mode by typing \"]\" and then do","category":"page"},{"location":"","page":"MetaQCD.jl: Metadynamics in Lattice QCD","title":"MetaQCD.jl: Metadynamics in Lattice QCD","text":"pkg> activate .","category":"page"},{"location":"","page":"MetaQCD.jl: Metadynamics in Lattice QCD","title":"MetaQCD.jl: Metadynamics in Lattice QCD","text":"Instantiate the project to install all the dependencies using the package manager:","category":"page"},{"location":"","page":"MetaQCD.jl: Metadynamics in Lattice QCD","title":"MetaQCD.jl: Metadynamics in Lattice QCD","text":"Pkg.instantiate()","category":"page"},{"location":"","page":"MetaQCD.jl: Metadynamics in Lattice QCD","title":"MetaQCD.jl: Metadynamics in Lattice QCD","text":"or","category":"page"},{"location":"","page":"MetaQCD.jl: Metadynamics in Lattice QCD","title":"MetaQCD.jl: Metadynamics in Lattice QCD","text":"pkg> instantiate","category":"page"},{"location":"","page":"MetaQCD.jl: Metadynamics in Lattice QCD","title":"MetaQCD.jl: Metadynamics in Lattice QCD","text":"If you want to use a GPU, make sure you not only have CUDA.jl or AMDGPU.jl installed, but also a fairly recent version of the CUDA Toolkit or ROCm.","category":"page"}]
}