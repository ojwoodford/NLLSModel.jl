# Generic function for testing a robustifier
function testrobustifier(kernel, costs, expected, firstorder=false)
    for (i, cost) in enumerate(costs)
        # Check the value is correct
        @test robustify(kernel, cost) ≈ expected[i]
        # Check the derivatives are correct
        ad = NLLSModel.autorobustifydcost(kernel, cost)
        if firstorder
            ad = (ad[1], ad[2], 0.0)
        end
        @test SVector(robustifydcost(kernel, cost)) ≈ SVector(ad)
        if isa(kernel, NLLSModel.AbstractAdaptiveRobustifier)
            # Check the derivatives w.r.t. the kernel parameters
            c, dc, d2c = robustifydkernel(kernel, cost)
            c_, dc_, d2c_ = NLLSModel.autorobustifydkernel(kernel, cost)
            @test c ≈ c_
            # @test dc ≈ dc_
            # @test d2c ≈ d2c_
        end
    end
end
