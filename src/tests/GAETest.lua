
--!strict

-- GAETest.lua

local GAE = loadstring(game:HttpGet("https://raw.githubusercontent.com/decorrupted/LuaPPO/refs/heads/main/src/GAE.lua"))()

local passed = 0
local failed = 0

local function test(name: string, callback: () -> ())
	local success, err = pcall(callback)

	if success then
		passed += 1
		print("[PASS] " .. name)
	else
		failed += 1
		warn("[FAIL] " .. name .. ": " .. tostring(err))
	end
end

local function expect(condition: boolean, message: string?)
	assert(condition, message or "Expected condition to be true")
end

local function expectEqual(
	actual: any,
	expected: any,
	message: string?
)
	assert(
		actual == expected,
		message or string.format(
			"Expected %s, got %s",
			tostring(expected),
			tostring(actual)
		)
	)
end

local function expectNear(
	actual: number,
	expected: number,
	epsilon: number?,
	message: string?
)
	local tolerance = epsilon or 1e-6

	assert(
		math.abs(actual - expected) <= tolerance,
		message or string.format(
			"Expected %.8f, got %.8f",
			expected,
			actual
		)
	)
end

local function expectError(callback: () -> ())
	local success = pcall(callback)
	expect(not success, "Expected an error, but none occurred")
end

local function makeExperience(
	reward: number,
	value: number,
	nextValue: number,
	terminated: boolean?,
	truncated: boolean?
)
	return {
		observation = {0.1, 0.2},
		action = {0.5, -0.5},

		reward = reward,
		value = value,
		logProb = -0.8,

		terminated = terminated or false,
		truncated = truncated or false,

		nextValue = nextValue,
	}
end

--------------------------------------------------
-- Test 1: Empty rollout
--------------------------------------------------

test("Empty rollout returns empty arrays", function()
	local result = GAE.compute({})

	expectEqual(#result.advantages, 0)
	expectEqual(#result.returns, 0)
end)

--------------------------------------------------
-- Test 2: Single normal transition
--------------------------------------------------

test("Single transition computes correct TD residual", function()
	local experiences = {
		makeExperience(1, 0.5, 0.8),
	}

	local result = GAE.compute(experiences, 0.9, 0.95)

	-- delta = 1 + 0.9 * 0.8 - 0.5 = 1.22
	expectNear(result.advantages[1], 1.22)
	expectNear(result.returns[1], 1.72)
end)

--------------------------------------------------
-- Test 3: True terminal ignores nextValue
--------------------------------------------------

test("Terminal transition does not bootstrap", function()
	local experiences = {
		makeExperience(1, 0.5, 100, true, false),
	}

	local result = GAE.compute(experiences, 0.9, 0.95)

	-- delta = reward - value = 0.5
	expectNear(result.advantages[1], 0.5)
	expectNear(result.returns[1], 1)
end)

--------------------------------------------------
-- Test 4: Truncation still bootstraps
--------------------------------------------------

test("Truncated transition bootstraps but stops recursion", function()
	local experiences = {
		makeExperience(1, 0.5, 2, false, true),
	}

	local result = GAE.compute(experiences, 0.9, 0.95)

	-- delta = 1 + 0.9 * 2 - 0.5 = 2.3
	expectNear(result.advantages[1], 2.3)
	expectNear(result.returns[1], 2.8)
end)

--------------------------------------------------
-- Test 5: Two-step advantage recursion
--------------------------------------------------

test("Two-step rollout uses GAE recursion", function()
	local experiences = {
		makeExperience(1, 0.5, 0.6),
		makeExperience(2, 0.6, 0, true, false),
	}

	local result = GAE.compute(experiences, 0.9, 0.8)

	-- Step 2 is terminal:
	-- delta2 = 2 - 0.6 = 1.4
	-- A2 = 1.4
	expectNear(result.advantages[2], 1.4)

	-- Step 1:
	-- delta1 = 1 + 0.9 * 0.6 - 0.5 = 1.04
	-- A1 = 1.04 + 0.9 * 0.8 * 1.4 = 2.048
	expectNear(result.advantages[1], 2.048)
	expectNear(result.returns[1], 2.548)
	expectNear(result.returns[2], 2)
end)

--------------------------------------------------
-- Test 6: Boundary prevents advantage leakage
--------------------------------------------------

test("Episode boundary stops recursive advantages", function()
	local experiences = {
		makeExperience(1, 0.5, 0.6, true, false),
		makeExperience(10, 0, 0, true, false),
	}

	local result = GAE.compute(experiences, 0.9, 0.95)

	-- The first terminal transition must not include
	-- the second episode's advantage.
	expectNear(result.advantages[1], 0.5)
	expectNear(result.advantages[2], 10)
end)

--------------------------------------------------
-- Test 7: Returns equal advantages plus values
--------------------------------------------------

test("Returns equal advantage plus value", function()
	local experiences = {
		makeExperience(1, 0.25, 0.5),
		makeExperience(2, 0.75, 0, true, false),
	}

	local result = GAE.compute(experiences, 0.99, 0.95)

	for i, experience in ipairs(experiences) do
		expectNear(
			result.returns[i],
			result.advantages[i] + experience.value
		)
	end
end)

--------------------------------------------------
-- Test 8: Default parameters
--------------------------------------------------

test("Default gamma and lambda are applied", function()
	local experiences = {
		makeExperience(1, 0.5, 0.8),
	}

	local defaultResult = GAE.compute(experiences)
	local explicitResult = GAE.compute(experiences, 0.99, 0.95)

	expectNear(
		defaultResult.advantages[1],
		explicitResult.advantages[1]
	)

	expectNear(
		defaultResult.returns[1],
		explicitResult.returns[1]
	)
end)

--------------------------------------------------
-- Test 9: Zero gamma
--------------------------------------------------

test("Zero gamma removes bootstrapping", function()
	local experiences = {
		makeExperience(2, 0.5, 100),
	}

	local result = GAE.compute(experiences, 0, 0.95)

	-- delta = reward - value = 1.5
	expectNear(result.advantages[1], 1.5)
	expectNear(result.returns[1], 2)
end)

--------------------------------------------------
-- Test 10: Zero lambda
--------------------------------------------------

test("Zero lambda removes recursive advantages", function()
	local experiences = {
		makeExperience(1, 0.5, 0.6),
		makeExperience(2, 0.6, 0, true, false),
	}

	local result = GAE.compute(experiences, 0.9, 0)

	expectNear(result.advantages[1], 1.04)
	expectNear(result.advantages[2], 1.4)
end)

--------------------------------------------------
-- Test 11: Reject invalid gamma
--------------------------------------------------

test("Invalid gamma is rejected", function()
	expectError(function()
		GAE.compute({}, -0.1, 0.95)
	end)

	expectError(function()
		GAE.compute({}, 1.1, 0.95)
	end)

	expectError(function()
		GAE.compute({}, 0 / 0, 0.95)
	end)
end)

--------------------------------------------------
-- Test 12: Reject invalid lambda
--------------------------------------------------

test("Invalid lambda is rejected", function()
	expectError(function()
		GAE.compute({}, 0.99, -0.1)
	end)

	expectError(function()
		GAE.compute({}, 0.99, 1.1)
	end)

	expectError(function()
		GAE.compute({}, 0.99, math.huge)
	end)
end)

--------------------------------------------------
-- Test 13: Reject invalid experience numbers
--------------------------------------------------

test("Invalid numeric values are rejected", function()
	local badReward = {
		makeExperience(1, 0.5, 0.6),
	}
	badReward[1].reward = 0 / 0

	expectError(function()
		GAE.compute(badReward)
	end)

	local badValue = {
		makeExperience(1, 0.5, 0.6),
	}
	badValue[1].value = math.huge

	expectError(function()
		GAE.compute(badValue)
	end)

	local badNextValue = {
		makeExperience(1, 0.5, 0.6),
	}
	badNextValue[1].nextValue = -math.huge

	expectError(function()
		GAE.compute(badNextValue)
	end)
end)

--------------------------------------------------
-- Test 14: Invalid episode flags
--------------------------------------------------

test("Invalid episode flags are rejected", function()
	local experiences = {
		makeExperience(1, 0.5, 0.6),
	}

	experiences[1].terminated = nil :: any

	expectError(function()
		GAE.compute(experiences)
	end)

	local invalidTruncation = {
		makeExperience(1, 0.5, 0.6),
	}

	invalidTruncation[1].truncated = 1 :: any

	expectError(function()
		GAE.compute(invalidTruncation)
	end)
end)

--------------------------------------------------
-- Test 15: Buffer integration
--------------------------------------------------

test("computeFromBuffer works with RolloutBuffer", function()
	local RolloutBuffer = require(script.Parent.RolloutBuffer)
	local buffer = RolloutBuffer.new(5)

	buffer:add(makeExperience(1, 0.5, 0.6))
	buffer:add(makeExperience(2, 0.6, 0, true, false))

	local direct = GAE.compute(buffer:getAll(), 0.9, 0.8)
	local integrated = GAE.computeFromBuffer(buffer, 0.9, 0.8)

	expectEqual(#integrated.advantages, 2)
	expectEqual(#integrated.returns, 2)

	for i = 1, 2 do
		expectNear(
			integrated.advantages[i],
			direct.advantages[i]
		)

		expectNear(
			integrated.returns[i],
			direct.returns[i]
		)
	end
end)

--------------------------------------------------
-- Results
--------------------------------------------------

print("--------------------------------")
print("GAE Test Results")
print("Passed: " .. passed .. "/15")
print("Failed: " .. failed .. "/15")
print("--------------------------------")

if failed > 0 then
	error(
		string.format(
			"GAE tests failed: %d passed, %d failed",
			passed,
			failed
		)
	)
end

print("All GAE tests passed!")
