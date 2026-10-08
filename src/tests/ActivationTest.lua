--!strict

--[[
	ActivationTest.lua

	Runnable tests for Activation.lua.

	Tests:
		1. ReLU construction
		2. Tanh construction
		3. ReLU positive values
		4. ReLU negative values
		5. ReLU zero
		6. ReLU mixed input
		7. ReLU backward
		8. Tanh forward
		9. Tanh backward
		10. Tanh derivative near zero
		11. Incorrect activation rejected
		12. Non-number input rejected
		13. Backward before forward rejected
		14. Gradient size mismatch rejected
		15. Clear removes cached input
]]

local Activation = loadstring(game:HttpGet("https://raw.githubusercontent.com/decorrupted/LuaPPO/refs/heads/main/src/Activation.lua"))()

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
			epsilon
		)
	end
end

--------------------------------------------------
-- 1. ReLU construction
--------------------------------------------------

test("ReLU construction", function()

	local activation = Activation.new("ReLU")

	assert(
		activation.activationType == "ReLU",
		"Activation type should be ReLU"
	)

end)

--------------------------------------------------
-- 2. Tanh construction
--------------------------------------------------

test("Tanh construction", function()

	local activation = Activation.new("Tanh")

	assert(
		activation.activationType == "Tanh",
		"Activation type should be Tanh"
	)

end)

--------------------------------------------------
-- 3. ReLU positive values
--------------------------------------------------

test("ReLU positive values", function()

	local activation = Activation.new("ReLU")

	local output = activation:forward({
		1,
		2,
		5.5,
	})

	expectVector(output, {
		1,
		2,
		5.5,
	})

end)

--------------------------------------------------
-- 4. ReLU negative values
--------------------------------------------------

test("ReLU negative values", function()

	local activation = Activation.new("ReLU")

	local output = activation:forward({
		-1,
		-2,
		-100,
	})

	expectVector(output, {
		0,
		0,
		0,
	})

end)

--------------------------------------------------
-- 5. ReLU zero
--------------------------------------------------

test("ReLU zero", function()

	local activation = Activation.new("ReLU")

	local output = activation:forward({
		0,
	})

	expectEqual(output[1], 0)

end)

--------------------------------------------------
-- 6. ReLU mixed input
--------------------------------------------------

test("ReLU mixed input", function()

	local activation = Activation.new("ReLU")

	local output = activation:forward({
		-2,
		0,
		3,
		-5,
		7,
	})

	expectVector(output, {
		0,
		0,
		3,
		0,
		7,
	})

end)

--------------------------------------------------
-- 7. ReLU backward
--------------------------------------------------

test("ReLU backward", function()

	local activation = Activation.new("ReLU")

	activation:forward({
		-2,
		0,
		3,
	})

	local gradient = activation:backward({
		10,
		20,
		30,
	})

	expectVector(gradient, {
		0,
		0,
		30,
	})

end)

--------------------------------------------------
-- 8. Tanh forward
--------------------------------------------------

test("Tanh forward", function()

	local activation = Activation.new("Tanh")

	local output = activation:forward({
		0,
		1,
		-1,
	})

	expectEqual(output[1], 0)

	expectEqual(
		output[2],
		math.tanh(1)
	)

	expectEqual(
		output[3],
		math.tanh(-1)
	)

end)

--------------------------------------------------
-- 9. Tanh backward
--------------------------------------------------

test("Tanh backward", function()

	local activation = Activation.new("Tanh")

	local input = {
		0,
		1,
		-1,
	}

	activation:forward(input)

	local gradient = activation:backward({
		1,
		1,
		1,
	})

	for i = 1, #input do

		local tanhX = math.tanh(input[i])
		local expected = 1 - tanhX * tanhX

		expectEqual(
			gradient[i],
			expected
		)

	end

end)

--------------------------------------------------
-- 10. Tanh derivative near zero
--------------------------------------------------

test("Tanh derivative near zero", function()

	local activation = Activation.new("Tanh")

	activation:forward({
		0,
	})

	local gradient = activation:backward({
		1,
	})

	-- d/dx tanh(x) at x = 0 is 1.
	expectEqual(
		gradient[1],
		1
	)

end)

--------------------------------------------------
-- 11. Incorrect activation rejected
--------------------------------------------------

test("Incorrect activation rejected", function()

	local success = pcall(function()

		Activation.new("Sigmoid" :: any)

	end)

	assert(
		success == false,
		"Invalid activation should throw an error"
	)

end)

--------------------------------------------------
-- 12. Non-number input rejected
--------------------------------------------------

test("Non-number input rejected", function()

	local activation = Activation.new("ReLU")

	local success = pcall(function()

		activation:forward({
			1,
			"hello" :: any,
			3,
		})

	end)

	assert(
		success == false,
		"Non-number input should throw an error"
	)

end)

--------------------------------------------------
-- 13. Backward before forward rejected
--------------------------------------------------

test("Backward before forward rejected", function()

	local activation = Activation.new("ReLU")

	local success = pcall(function()

		activation:backward({
			1,
			2,
		})

	end)

	assert(
		success == false,
		"Backward before forward should throw an error"
	)

end)

--------------------------------------------------
-- 14. Gradient size mismatch rejected
--------------------------------------------------

test("Gradient size mismatch rejected", function()

	local activation = Activation.new("ReLU")

	activation:forward({
		1,
		2,
		3,
	})

	local success = pcall(function()

		activation:backward({
			1,
			2,
		})

	end)

	assert(
		success == false,
		"Incorrect gradient size should throw an error"
	)

end)

--------------------------------------------------
-- 15. Clear removes cached input
--------------------------------------------------

test("Clear removes cached input", function()

	local activation = Activation.new("ReLU")

	activation:forward({
		1,
		2,
	})

	assert(
		activation.lastInput ~= nil,
		"Forward should cache input"
	)

	activation:clear()

	assert(
		activation.lastInput == nil,
		"Clear should remove cached input"
	)

end)

--------------------------------------------------
-- Results
--------------------------------------------------

print("")
print("==============================")
print("Activation Tests")
print("==============================")
print("Passed:", passed)
print("Failed:", failed)
print("Total: ", passed + failed)
print("==============================")

assert(
	failed == 0,
	string.format(
		"%d Activation tests failed",
		failed
	)
)

print("ALL ACTIVATION TESTS PASSED")