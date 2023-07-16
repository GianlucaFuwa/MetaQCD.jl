module IOModule
    using JLD2
    using LinearAlgebra
    using ..VerbosePrint

    export SaveConfigs, loadU_bridge!, loadU_jld!, save_gaugefield, saveU_bridge, saveU_jld

    # struct BMWFormat end
    struct BridgeFormat end
    struct JLD2Format end

    # include("bmw_format.jl")
    include("bridge_format.jl")
    include("jld2_format.jl")

    struct SaveConfigs{T}
        saveU_format::Union{Nothing, String}
        saveU_dir::String
        saveU_every::Int64
        itrjsavecount::Base.RefValue{Int64}

        function SaveConfigs(saveU_format, saveU_dir, saveU_every, vp)
            itrjsavecount = Base.RefValue{Int64}(0)

            if saveU_format == "bridge"
                T = BridgeFormat
            elseif saveU_format == "jld" || saveU_format == "jld2"
                T = JLD2Format
            elseif saveU_format === nothing
                T = Nothing
            else
                error("saveU format $saveU_format not supported")
            end

            if T !== Nothing
                println_verbose1(vp, "\t>> Save config every $(saveU_every) trajectory!")
            end

            return new{T}(saveU_format, saveU_dir, saveU_every, itrjsavecount)
        end
    end

    function save_gaugefield(save_configs::SaveConfigs{T}, U, itrj) where {T}
        T === Nothing && return nothing

        if itrj % save_configs.saveU_every == 0
            save_configs.itrjsavecount[] += 1
            itrjstring = lpad(itrj, 8, "0")

            if T == JLD2Format
                filename = save_configs.saveU_dir * "/conf_$(itrjstring).jld2"
                saveU_jld(U, filename)
            elseif T == BridgeFormat
                filename = save_configs.saveU_dir * "/conf_$(itrjstring).txt"
                saveU_bridge(U, filename)
            end
        end

        return nothing
    end

end
