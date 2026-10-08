--!strict

--[[
	NetworkTest.lua

	Runnable tests for Network.lua.

	Tests:
		1. Network construction
		2. Correct layer count
		3. Parameter count
		4. Forward output size
		5. Forward pass with known parameters
		6. Activation layer in forward pass
		7. Backward output size
		8. Backward through multiple layers
		9. Known backward gradient
		10. zeroGrad
		11. step
		12. clear
		13. Invalid layer configuration rejected
		14. Dimension mismatch rejected
		15. Backward before forward rejected
]]

local Network = loadstring(game:HttpGet("https://raw.githubusercontent.com/decorrupted/LuaPPO/refs/heads/main/src/Network.lua"))()

local passed = 0
local failed = 0

--------------------------------------------------
-- Test helpers
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

local function expectVector(
	actual: {number},
	expected: {number},
	epsilon: number?
)
	local tolerance = epsilon or 1e-6

	assert(
		#actual == #expected,
		string.format(
			"Expected vector size %d, got %d",
			#expected,
			#actual
		)
	)

	for i = 1, #expected do

		expectEqual(
			actual[i],
			expected[i],
			tolerance
		)

	end
end

--------------------------------------------------
-- 1. Network construction
--------------------------------------------------

test("Network construction", function()

	local network = Network.new({
		{inputSize = 2, outputSize = 4},
		{activation = "Tanh"},
		{inputSize = 4, outputSize = 2},
	})

	assert(
		network ~= nil,
		"Network should be created"
	)

end)

--------------------------------------------------
-- 2. Correct layer count
--------------------------------------------------

test("Correct layer count", function()

	local network = Network.new({
		{inputSize = 3, outputSize = 5},
		{activation = "ReLU"},
		{inputSize = 5, outputSize = 2},
	})

	assert(
		#network.layers == 3,
		string.format(
			"Expected 3 layers, got %d",
			#network.layers
		)
	)

end)

--------------------------------------------------
-- 3. Parameter count
--------------------------------------------------

test("Parameter count", function()

	local network = Network.new({
		{inputSize = 3, outputSize = 4},
		{activation = "Tanh"},
		{inputSize = 4, outputSize = 2},
	})

	-- First Linear:
	-- 3 * 4 weights + 4 biases = 16
	--
	-- Second Linear:
	-- 4 * 2 weights + 2 biases = 10
	--
	-- Total = 26

	assert(
		network:getParameterCount() == 26,
		string.format(
			"Expected 26 parameters, got %d",
			network:getParameterCount()
		)
	)

end)

--------------------------------------------------
-- 4. Forward output size
--------------------------------------------------

test("Forward output size", function()

	local network = Network.new({
		{inputSize = 4, outputSize = 8},
		{activation = "Tanh"},
		{inputSize = 8, outputSize = 3},
	})

	local output = network:forward({
		1,
		2,
		3,
		4,
	})

	assert(
		#output == 3,
		string.format(
			"Expected 3 outputs, got %d",
			#output
		)
	)

end)

--------------------------------------------------
-- 5. Forward pass with known parameters
--------------------------------------------------

test("Forward pass with known parameters", function()

	local network = Network.new({
		{inputSize = 2, outputSize = 2},
	})

	local layer = network.layers[1]

	-- W =
	-- [ 1  2 ]
	-- [ 3  4 ]
	--
	-- b = [1, 2]

	layer.weights.data[1][1] = 1
	layer.weights.data[1][2] = 2

	layer.weights.data[2][1] = 3
	layer.weights.data[2][2] = 4

	layer.bias[1] = 1
	layer.bias[2] = 2

	local output = network:forward({
		5,
		6,
	})

	-- y1 = 1*5 + 2*6 + 1 = 18
	-- y2 = 3*5 + 4*6 + 2 = 41

	expectVector(output, {
		18,
		41,
	})

end)

--------------------------------------------------
-- 6. Activation layer in forward pass
--------------------------------------------------

test("Activation layer in forward pass", function()

	local network = Network.new({
		{inputSize = 2, outputSize = 2},
		{activation = "ReLU"},
	})

	local layer = network.layers[1]

	layer.weights.data[1][1] = 1
	layer.weights.data[1][2] = 0

	layer.weights.data[2][1] = 0
	layer.weights.data[2][2] = 1

	layer.bias[1] = 0
	layer.bias[2] = 0

	local output = network:forward({
		-5,
		3,
	})

	expectVector(output, {
		0,
		3,
	})

end)

--------------------------------------------------
-- 7. Backward output size
--------------------------------------------------

test("Backward output size", function()

	local network = Network.new({
		{inputSize = 4, outputSize = 6},
		{activation = "Tanh"},
		{inputSize = 6, outputSize = 2},
	})

	network:forward({
		1,
		2,
		3,
		4,
	})

	local gradient = network:backward({
		1,
		1,
	})

	assert(
		#gradient == 4,
		string.format(
			"Expected gradient size 4, got %d",
			#gradient
		)
	)

end)

--------------------------------------------------
-- 8. Backward through multiple layers
--------------------------------------------------

test("Backward through multiple layers", function()

	local network = Network.new({
		{inputSize = 2, outputSize = 3},
		{activation = "ReLU"},
		{inputSize = 3, outputSize = 2},
	})

	local output = network:forward({
		1,
		2,
	})

	assert(
		#output == 2,
		"Forward pass failed"
	)

	local gradient = network:backward({
		1,
		1,
	})

	assert(
		#gradient == 2,
		"Backward pass did not reach input"
	)

end)

--------------------------------------------------
-- 9. Known backward gradient
--------------------------------------------------

test("Known backward gradient", function()

	local network = Network.new({
		{inputSize = 2, outputSize = 2},
	})

	local layer = network.layers[1]

	-- W =
	-- [1 2]
	-- [3 4]

	layer.weights.data[1][1] = 1
	layer.weights.data[1][2] = 2

	layer.weights.data[2][1] = 3
	layer.weights.data[2][2] = 4

	network:forward({
		5,
		6,
	})

	local gradient = network:backward({
		10,
		20,
	})

	-- W^T * gradient
	--
	-- dx1 = 1*10 + 3*20 = 70
	-- dx2 = 2*10 + 4*20 = 100

	expectVector(gradient, {
		70,
		100,
	})

end)

--------------------------------------------------
-- 10. zeroGrad
--------------------------------------------------

test("zeroGrad", function()

	local network = Network.new({
		{inputSize = 2, outputSize = 2},
	})

	local layer = network.layers[1]

	network:forward({
		1,
		2,
	})

	network:backward({
		3,
		4,
	})

	assert(
		layer.gradBias[1] ~= 0,
		"Gradient should exist before zeroGrad"
	)

	network:zeroGrad()

	for i = 1, 2 do

		assert(
			layer.gradBias[i] == 0,
			"Bias gradient was not cleared"
		)

		for j = 1, 2 do

			assert(
				layer.gradWeights.data[i][j] == 0,
				"Weight gradient was not cleared"
			)

		end
	end

end)

--------------------------------------------------
-- 11. step
--------------------------------------------------

test("step", function()

	local network = Network.new({
		{inputSize = 1, outputSize = 1},
	})

	local layer = network.layers[1]

	layer.weights.data[1][1] = 10
	layer.bias[1] = 5

	network:forward({
		2,
	})

	network:backward({
		3,
	})

	network:step(0.1)

	-- Weight gradient = 3 * 2 = 6
	-- New weight = 10 - 0.1*6 = 9.4
	--
	-- Bias gradient = 3
	-- New bias = 5 - 0.1*3 = 4.7

	expectEqual(
		layer.weights.data[1][1],
		9.4
	)

	expectEqual(
		layer.bias[1],
		4.7
	)

end)

--------------------------------------------------
-- 12. clear
--------------------------------------------------

test("clear", function()

	local network = Network.new({
		{inputSize = 2, outputSize = 2},
		{activation = "Tanh"},
	})

	network:forward({
		1,
		2,
	})

	assert(
		network.lastOutput ~= nil,
		"Network should store last output"
	)

	network:clear()

	assert(
		network.lastOutput == nil,
		"Network output cache was not cleared"
	)

	local linearLayer = network.layers[1]
	local activationLayer = network.layers[2]

	assert(
		linearLayer.lastInput == nil,
		"Linear input cache was not cleared"
	)

	assert(
		activationLayer.lastInput == nil,
		"Activation input cache was not cleared"
	)

end)

--------------------------------------------------
-- 13. Invalid layer configuration rejected
--------------------------------------------------

test("Invalid layer configuration rejected", function()

	local success = pcall(function()

		Network.new({
			{
				foo = "bar",
			} :: any,
		})

	end)

	assert(
		success == false,
		"Invalid configuration should throw an error"
	)

end)

--------------------------------------------------
-- 14. Dimension mismatch rejected
--------------------------------------------------

test("Dimension mismatch rejected", function()

	local success = pcall(function()

		Network.new({
			{inputSize = 2, outputSize = 4},
			{inputSize = 7, outputSize = 3},
		})

	end)

	assert(
		success == false,
		"Dimension mismatch should throw an error"
	)

end)

--------------------------------------------------
-- 15. Backward before forward rejected
--------------------------------------------------

test("Backward before forward rejected", function()

	local network = Network.new({
		{inputSize = 2, outputSize = 2},
	})

	local success = pcall(function()

		network:backward({
			1,
			1,
		})

	end)

	assert(
		success == false,
		"Backward before forward should throw an error"
	)

end)

--------------------------------------------------
-- Results
--------------------------------------------------

print("")
print("==============================")
print("Network Tests")
print("==============================")
print("Passed:", passed)
print("Failed:", failed)
print("Total: ", passed + failed)
print("==============================")

assert(
	failed == 0,
	string.format(
		"%d Network tests failed",
		failed
	)
)

print("ALL NETWORK TESTS PASSED")