--!strict

--[[
	LinearTest.lua

	Runnable test for Linear.lua.

	Expected hierarchy:

		LinearTest.lua
		├── Matrix.lua
		└── Linear.lua

	Run this from Roblox Studio or require it
	from a Script/Command Bar.
]]

local folder = script.Parent

local Linear = loadstring(game:HttpGet("https://raw.githubusercontent.com/decorrupted/LuaPPO/refs/heads/main/src/Linear.lua"))()

--------------------------------------------------
-- Test utilities
--------------------------------------------------

local passed = 0
local failed = 0

local function test(
	name: string,
	fn: () -> ()
)
	local success, errorMessage = pcall(fn)

	if success then
		passed += 1
		print("✓ PASS:", name)
	else
		failed += 1
		warn("✗ FAIL:", name)
		warn("  ", errorMessage)
	end
end

local function assertEqual(
	actual: number,
	expected: number,
	tolerance: number?,
	message: string?
)
	local epsilon = tolerance or 1e-6

	assert(
		math.abs(actual - expected) <= epsilon,
		message
			or string.format(
				"Expected %f, got %f",
				expected,
				actual
			)
	)
end

local function assertVectorSize(
	vector: {number},
	expected: number
)
	assert(
		#vector == expected,
		string.format(
			"Expected vector size %d, got %d",
			expected,
			#vector
		)
	)
end

--------------------------------------------------
-- Tests
--------------------------------------------------

print("")
print("======================================")
print(" Linear.lua Tests")
print("======================================")
print("")

--------------------------------------------------
-- 1. Construction
--------------------------------------------------

test("Creates a Linear layer", function()

	local layer = Linear.new(3, 2)

	assert(layer.inputSize == 3)
	assert(layer.outputSize == 2)

end)

--------------------------------------------------
-- 2. Parameter dimensions
--------------------------------------------------

test("Weight dimensions are correct", function()

	local layer = Linear.new(3, 2)

	assert(layer.weights.rows == 2)
	assert(layer.weights.cols == 3)

end)

test("Bias dimensions are correct", function()

	local layer = Linear.new(3, 2)

	assert(#layer.bias == 2)

end)

--------------------------------------------------
-- 3. Parameter count
--------------------------------------------------

test("Parameter count is correct", function()

	local layer = Linear.new(3, 2)

	-- 3 inputs × 2 outputs = 6 weights
	-- + 2 biases
	-- = 8 parameters

	assert(
		layer:getParameterCount() == 8,
		"Expected 8 parameters"
	)

end)

--------------------------------------------------
-- 4. Forward pass
--------------------------------------------------

test("Forward pass produces correct output size", function()

	local layer = Linear.new(3, 2)

	local output = layer:forward({
		1,
		2,
		3,
	})

	assertVectorSize(output, 2)

end)

--------------------------------------------------
-- 5. Forward pass with known parameters
--------------------------------------------------

test("Forward pass calculates W*x+b correctly", function()

	local layer = Linear.new(2, 2)

	-- Manually set:

	-- W =
	-- [ 1  2 ]
	-- [ 3  4 ]

	layer.weights.data[1][1] = 1
	layer.weights.data[1][2] = 2

	layer.weights.data[2][1] = 3
	layer.weights.data[2][2] = 4

	-- b = [5, 6]

	layer.bias[1] = 5
	layer.bias[2] = 6

	-- x = [2, 3]

	local output = layer:forward({
		2,
		3,
	})

	-- y1 = 1*2 + 2*3 + 5
	--    = 13
	--
	-- y2 = 3*2 + 4*3 + 6
	--    = 24

	assertEqual(output[1], 13)
	assertEqual(output[2], 24)

end)

--------------------------------------------------
-- 6. Invalid input size
--------------------------------------------------

test("Rejects incorrect input size", function()

	local layer = Linear.new(3, 2)

	local success = pcall(function()

		layer:forward({
			1,
			2,
		})

	end)

	assert(
		not success,
		"Layer should reject incorrect input size"
	)

end)

--------------------------------------------------
-- 7. Invalid input type
--------------------------------------------------

test("Rejects non-number input", function()

	local layer = Linear.new(3, 2)

	local success = pcall(function()

		layer:forward({
			1,
			"hello" :: any,
			3,
		})

	end)

	assert(
		not success,
		"Layer should reject non-number input"
	)

end)

--------------------------------------------------
-- 8. Backward pass
--------------------------------------------------

test("Backward pass returns correct gradient size", function()

	local layer = Linear.new(3, 2)

	layer:forward({
		1,
		2,
		3,
	})

	local gradientInput = layer:backward({
		1,
		1,
	})

	assertVectorSize(
		gradientInput,
		3
	)

end)

--------------------------------------------------
-- 9. Backward pass gradient calculation
--------------------------------------------------

test("Backward pass calculates input gradients", function()

	local layer = Linear.new(2, 2)

	-- W =
	-- [ 1  2 ]
	-- [ 3  4 ]

	layer.weights.data[1][1] = 1
	layer.weights.data[1][2] = 2

	layer.weights.data[2][1] = 3
	layer.weights.data[2][2] = 4

	layer.bias[1] = 0
	layer.bias[2] = 0

	layer:forward({
		5,
		6,
	})

	-- dL/dY = [1, 2]

	local gradientInput = layer:backward({
		1,
		2,
	})

	-- dL/dX = Wᵀ * dL/dY
	--
	-- [1 3] [1] = [7]
	-- [2 4] [2]   [10]

	assertEqual(
		gradientInput[1],
		7
	)

	assertEqual(
		gradientInput[2],
		10
	)

end)

--------------------------------------------------
-- 10. Weight gradients
--------------------------------------------------

test("Weight gradients are correct", function()

	local layer = Linear.new(2, 2)

	layer.weights.data[1][1] = 1
	layer.weights.data[1][2] = 2

	layer.weights.data[2][1] = 3
	layer.weights.data[2][2] = 4

	layer:forward({
		5,
		6,
	})

	-- dL/dY = [1, 2]

	layer:backward({
		1,
		2,
	})

	-- dL/dW =
	--
	-- [1*5  1*6]
	-- [2*5  2*6]
	--
	-- [5  6]
	-- [10 12]

	assertEqual(
		layer.gradWeights.data[1][1],
		5
	)

	assertEqual(
		layer.gradWeights.data[1][2],
		6
	)

	assertEqual(
		layer.gradWeights.data[2][1],
		10
	)

	assertEqual(
		layer.gradWeights.data[2][2],
		12
	)

end)

--------------------------------------------------
-- 11. Bias gradients
--------------------------------------------------

test("Bias gradients are correct", function()

	local layer = Linear.new(2, 2)

	layer:forward({
		5,
		6,
	})

	layer:backward({
		1,
		2,
	})

	assertEqual(
		layer.gradBias[1],
		1
	)

	assertEqual(
		layer.gradBias[2],
		2
	)

end)

--------------------------------------------------
-- 12. Zero gradients
--------------------------------------------------

test("zeroGrad clears gradients", function()

	local layer = Linear.new(2, 2)

	layer:forward({
		5,
		6,
	})

	layer:backward({
		1,
		2,
	})

	layer:zeroGrad()

	for i = 1, layer.outputSize do

		assertEqual(
			layer.gradBias[i],
			0
		)

		for j = 1, layer.inputSize do

			assertEqual(
				layer.gradWeights.data[i][j],
				0
			)

		end
	end

end)

--------------------------------------------------
-- 13. SGD step
--------------------------------------------------

test("Parameter update works", function()

	local layer = Linear.new(1, 1)

	layer.weights.data[1][1] = 10
	layer.bias[1] = 5

	layer:forward({
		2,
	})

	layer:backward({
		3,
	})

	-- gradient weight = 3 * 2 = 6
	-- gradient bias   = 3
	--
	-- learning rate = 0.1
	--
	-- weight = 10 - 0.1*6 = 9.4
	-- bias   = 5  - 0.1*3 = 4.7

	layer:step(0.1)

	assertEqual(
		layer.weights.data[1][1],
		9.4
	)

	assertEqual(
		layer.bias[1],
		4.7
	)

end)

--------------------------------------------------
-- 14. Backward before forward
--------------------------------------------------

test("Rejects backward before forward", function()

	local layer = Linear.new(2, 2)

	local success = pcall(function()

		layer:backward({
			1,
			1,
		})

	end)

	assert(
		not success,
		"Backward should require a previous forward pass"
	)

end)

--------------------------------------------------
-- Results
--------------------------------------------------

print("")
print("======================================")
print(" Results")
print("======================================")

print("Passed:", passed)
print("Failed:", failed)
print("Total: ", passed + failed)

if failed == 0 then
	print("")
	print("✓ ALL LINEAR TESTS PASSED")
else
	warn("")
	warn("✗ SOME TESTS FAILED")
end

print("")