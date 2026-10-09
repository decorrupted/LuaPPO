
--!strict

--[[
	AdamTest.lua

	15 runnable tests for Adam.lua.

	Tests:
		1. Default configuration
		2. Custom configuration
		3. First weight update
		4. First bias update
		5. Bias correction
		6. Second optimizer step
		7. Moment estimates persist
		8. Multiple weights update
		9. Multiple layers update
		10. Learning rate can change
		11. Zero gradients leave parameters unchanged
		12. Reset clears optimizer history
		13. Invalid learning rate rejected
		14. Invalid beta rejected
		15. Invalid epsilon rejected
]]

local Network = loadstring(game:HttpGet("https://raw.githubusercontent.com/decorrupted/LuaPPO/refs/heads/main/src/Network.lua"))()
local Adam = loadstring(game:HttpGet("https://raw.githubusercontent.com/decorrupted/LuaPPO/refs/heads/main/src/Adam.lua"))()


local passed = 0
local failed = 0

--------------------------------------------------
-- Helpers
--------------------------------------------------

local function test(
	name: string,
	callback: () -> ()
)
	local success, errorMessage = pcall(callback)

	if success then
		passed += 1
		print("PASS:", name)
	else
		failed += 1
		warn("FAIL:", name)
		warn("     ", errorMessage)
	end
end

local function expectEqual(
	actual: number,
	expected: number,
	epsilon: number?
)
	local tolerance = epsilon or 1e-6

	assert(
		math.abs(actual - expected) <= tolerance,
		string.format(
			"Expected %.10f, got %.10f",
			expected,
			actual
		)
	)
end

local function makeNetwork(
	inputSize: number?,
	outputSize: number?
)
	return Network.new({
		{
			inputSize = inputSize or 1,
			outputSize = outputSize or 1,
		},
	})
end

local function prepareGradient(
	network: any,
	weightGradient: number,
	biasGradient: number
)
	local layer = network.layers[1]

	network:zeroGrad()

	layer.gradWeights.data[1][1] = weightGradient
	layer.gradBias[1] = biasGradient
end

--------------------------------------------------
-- 1. Default configuration
--------------------------------------------------

test("Default configuration", function()
	local network = makeNetwork()
	local optimizer = Adam.new(network)

	expectEqual(optimizer.learningRate, 0.001)
	expectEqual(optimizer.beta1, 0.9)
	expectEqual(optimizer.beta2, 0.999)
	expectEqual(optimizer.epsilon, 1e-8)
	expectEqual(optimizer.timeStep, 0)
end)

--------------------------------------------------
-- 2. Custom configuration
--------------------------------------------------

test("Custom configuration", function()
	local network = makeNetwork()

	local optimizer = Adam.new(
		network,
		0.01,
		0.8,
		0.99,
		1e-6
	)

	expectEqual(optimizer.learningRate, 0.01)
	expectEqual(optimizer.beta1, 0.8)
	expectEqual(optimizer.beta2, 0.99)
	expectEqual(optimizer.epsilon, 1e-6)
end)

--------------------------------------------------
-- 3. First weight update
--------------------------------------------------

test("First weight update", function()
	local network = makeNetwork()
	local layer = network.layers[1]

	layer.weights.data[1][1] = 2

	local optimizer = Adam.new(
		network,
		0.01
	)

	prepareGradient(network, 3, 0)

	optimizer:step()

	-- On step one, bias correction makes mHat = g
	-- and vHat = g^2.
	--
	-- Update = lr * g / (abs(g) + epsilon)

	local expected = 2
		- 0.01 * 3 / (3 + 1e-8)

	expectEqual(
		layer.weights.data[1][1],
		expected
	)
end)

--------------------------------------------------
-- 4. First bias update
--------------------------------------------------

test("First bias update", function()
	local network = makeNetwork()
	local layer = network.layers[1]

	layer.bias[1] = 5

	local optimizer = Adam.new(
		network,
		0.01
	)

	prepareGradient(network, 0, -2)

	optimizer:step()

	local expected = 5
		- 0.01 * (-2) / (2 + 1e-8)

	expectEqual(
		layer.bias[1],
		expected
	)
end)

--------------------------------------------------
-- 5. Bias correction
--------------------------------------------------

test("Bias correction", function()
	local network = makeNetwork()
	local layer = network.layers[1]

	layer.weights.data[1][1] = 0

	local optimizer = Adam.new(
		network,
		0.01
	)

	prepareGradient(network, 2, 0)
	optimizer:step()

	-- Despite the initial moment estimates being
	-- small, bias correction makes the first update
	-- approximately equal to the learning rate.

	expectEqual(
		layer.weights.data[1][1],
		-0.01 * 2 / (2 + 1e-8)
	)

	expectEqual(optimizer.timeStep, 1)
end)

--------------------------------------------------
-- 6. Second optimizer step
--------------------------------------------------

test("Second optimizer step", function()
	local network = makeNetwork()
	local layer = network.layers[1]

	layer.weights.data[1][1] = 1

	local optimizer = Adam.new(
		network,
		0.01
	)

	prepareGradient(network, 1, 0)
	optimizer:step()

	local firstWeight = layer.weights.data[1][1]

	prepareGradient(network, 1, 0)
	optimizer:step()

	assert(
		layer.weights.data[1][1] < firstWeight,
		"Positive gradient should continue decreasing the weight"
	)

	expectEqual(optimizer.timeStep, 2)
end)

--------------------------------------------------
-- 7. Moment estimates persist
--------------------------------------------------

test("Moment estimates persist", function()
	local network = makeNetwork()
	local optimizer = Adam.new(network)

	local layer = network.layers[1]
	local state = optimizer.stateByLayer[layer]

	prepareGradient(network, 2, 0)
	optimizer:step()

	assert(
		state.mWeights[1][1] ~= 0,
		"First moment should be updated"
	)

	assert(
		state.vWeights[1][1] ~= 0,
		"Second moment should be updated"
	)

	local firstMoment = state.mWeights[1][1]

	prepareGradient(network, 0, 0)
	optimizer:step()

	expectEqual(
		state.mWeights[1][1],
		0.9 * firstMoment
	)
end)

--------------------------------------------------
-- 8. Multiple weights update
--------------------------------------------------

test("Multiple weights update", function()
	local network = makeNetwork(2, 2)
	local layer = network.layers[1]

	layer.weights.data[1][1] = 1
	layer.weights.data[1][2] = 2
	layer.weights.data[2][1] = 3
	layer.weights.data[2][2] = 4

	local optimizer = Adam.new(network, 0.01)

	network:zeroGrad()

	layer.gradWeights.data[1][1] = 1
	layer.gradWeights.data[1][2] = -2
	layer.gradWeights.data[2][1] = 3
	layer.gradWeights.data[2][2] = -4

	optimizer:step()

	assert(layer.weights.data[1][1] < 1)
	assert(layer.weights.data[1][2] > 2)
	assert(layer.weights.data[2][1] < 3)
	assert(layer.weights.data[2][2] > 4)
end)

--------------------------------------------------
-- 9. Multiple layers update
--------------------------------------------------

test("Multiple layers update", function()
	local network = Network.new({
		{inputSize = 2, outputSize = 3},
		{activation = "Tanh"},
		{inputSize = 3, outputSize = 1},
	})

	local optimizer = Adam.new(network, 0.01)

	local firstLayer = network.layers[1]
	local secondLayer = network.layers[3]

	firstLayer.weights.data[1][1] = 1
	secondLayer.weights.data[1][1] = 1

	network:zeroGrad()

	firstLayer.gradWeights.data[1][1] = 1
	secondLayer.gradWeights.data[1][1] = 1

	optimizer:step()

	assert(firstLayer.weights.data[1][1] < 1)
	assert(secondLayer.weights.data[1][1] < 1)
end)

--------------------------------------------------
-- 10. Learning rate can change
--------------------------------------------------

test("Learning rate can change", function()
	local network = makeNetwork()
	local optimizer = Adam.new(network)

	optimizer:setLearningRate(0.005)

	expectEqual(
		optimizer.learningRate,
		0.005
	)
end)

--------------------------------------------------
-- 11. Zero gradients leave parameters unchanged
--------------------------------------------------

test("Zero gradients leave parameters unchanged", function()
	local network = makeNetwork()
	local layer = network.layers[1]

	layer.weights.data[1][1] = 2
	layer.bias[1] = 3

	local optimizer = Adam.new(network, 0.01)

	prepareGradient(network, 0, 0)
	optimizer:step()

	expectEqual(
		layer.weights.data[1][1],
		2
	)

	expectEqual(
		layer.bias[1],
		3
	)
end)

--------------------------------------------------
-- 12. Reset clears optimizer history
--------------------------------------------------

test("Reset clears optimizer history", function()
	local network = makeNetwork()
	local optimizer = Adam.new(network)

	local layer = network.layers[1]

	prepareGradient(network, 2, 3)
	optimizer:step()

	optimizer:reset()

	expectEqual(optimizer.timeStep, 0)

	local state = optimizer.stateByLayer[layer]

	expectEqual(state.mWeights[1][1], 0)
	expectEqual(state.vWeights[1][1], 0)
	expectEqual(state.mBias[1], 0)
	expectEqual(state.vBias[1], 0)
end)

--------------------------------------------------
-- 13. Invalid learning rate rejected
--------------------------------------------------

test("Invalid learning rate rejected", function()
	local network = makeNetwork()

	local success = pcall(function()
		Adam.new(network, -0.01)
	end)

	assert(not success, "Negative learning rate should be rejected")
end)

--------------------------------------------------
-- 14. Invalid beta rejected
--------------------------------------------------

test("Invalid beta rejected", function()
	local network = makeNetwork()

	local success = pcall(function()
		Adam.new(network, 0.001, 1.0)
	end)

	assert(not success, "beta1 = 1 should be rejected")
end)

--------------------------------------------------
-- 15. Invalid epsilon rejected
--------------------------------------------------

test("Invalid epsilon rejected", function()
	local network = makeNetwork()

	local success = pcall(function()
		Adam.new(network, 0.001, 0.9, 0.999, 0)
	end)

	assert(not success, "Zero epsilon should be rejected")
end)

--------------------------------------------------
-- Results
--------------------------------------------------

print("")
print("==============================")
print("Adam Tests")
print("==============================")
print("Passed:", passed)
print("Failed:", failed)
print("Total: ", passed + failed)
print("==============================")

assert(
	failed == 0,
	string.format(
		"%d Adam tests failed",
		failed
	)
)

print("ALL ADAM TESTS PASSED")
