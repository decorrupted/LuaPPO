--!strict

-- RolloutBufferTest.lua

local RolloutBuffer = loadstring(game:HttpGet("https://raw.githubusercontent.com/decorrupted/LuaPPO/refs/heads/main/src/RolloutBuffer.lua"))()

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

local function expectError(callback: () -> ())
	local success = pcall(callback)
	expect(not success, "Expected an error, but none occurred")
end

local function makeExperience(
	reward: number?,
	observation: {number}?,
	action: {number}?
)
	return {
		observation = observation or {0.1, 0.2, 0.3},
		action = action or {0.5, -0.5},
		reward = reward or 1,
		value = 0.4,
		logProb = -0.8,
		terminated = false,
		truncated = false,
		nextValue = 0.45,
	}
end

--------------------------------------------------
-- Test 1: Constructor
--------------------------------------------------

test("Constructor creates an empty buffer", function()
	local buffer = RolloutBuffer.new(10)

	expectEqual(buffer:getSize(), 0)
	expectEqual(buffer.capacity, 10)
	expect(not buffer:isFull())
end)

--------------------------------------------------
-- Test 2: Invalid capacity
--------------------------------------------------

test("Constructor rejects invalid capacity", function()
	expectError(function()
		RolloutBuffer.new(0)
	end)

	expectError(function()
		RolloutBuffer.new(-1)
	end)

	expectError(function()
		RolloutBuffer.new(1.5)
	end)
end)

--------------------------------------------------
-- Test 3: Add one experience
--------------------------------------------------

test("Add stores an experience", function()
	local buffer = RolloutBuffer.new(10)

	buffer:add(makeExperience())

	expectEqual(buffer:getSize(), 1)
end)

--------------------------------------------------
-- Test 4: Retrieve stored data
--------------------------------------------------

test("Get returns the correct experience", function()
	local buffer = RolloutBuffer.new(10)
	local experience = makeExperience(2.5)

	buffer:add(experience)

	local result = buffer:get(1)

	expectEqual(result.reward, 2.5)
	expectEqual(result.value, 0.4)
	expectEqual(result.logProb, -0.8)
	expectEqual(result.nextValue, 0.45)
	expectEqual(result.observation[2], 0.2)
	expectEqual(result.action[1], 0.5)
end)

--------------------------------------------------
-- Test 5: Multiple experiences preserve order
--------------------------------------------------

test("Multiple experiences preserve insertion order", function()
	local buffer = RolloutBuffer.new(5)

	buffer:add(makeExperience(1))
	buffer:add(makeExperience(2))
	buffer:add(makeExperience(3))

	expectEqual(buffer:getSize(), 3)
	expectEqual(buffer:get(1).reward, 1)
	expectEqual(buffer:get(2).reward, 2)
	expectEqual(buffer:get(3).reward, 3)
end)

--------------------------------------------------
-- Test 6: Observation is copied on insertion
--------------------------------------------------

test("Observation is copied when added", function()
	local buffer = RolloutBuffer.new(5)
	local observation = {1, 2, 3}
	local experience = makeExperience(1, observation)

	buffer:add(experience)

	observation[1] = 999

	expectEqual(buffer:get(1).observation[1], 1)
end)

--------------------------------------------------
-- Test 7: Action is copied on insertion
--------------------------------------------------

test("Action is copied when added", function()
	local buffer = RolloutBuffer.new(5)
	local action = {0.2, -0.4}

	buffer:add(makeExperience(1, nil, action))

	action[1] = 999

	expectEqual(buffer:get(1).action[1], 0.2)
end)

--------------------------------------------------
-- Test 8: Retrieved vectors are independent copies
--------------------------------------------------

test("Modifying retrieved data does not change buffer", function()
	local buffer = RolloutBuffer.new(5)

	buffer:add(makeExperience())

	local result = buffer:get(1)
	result.observation[1] = 999
	result.action[1] = 999

	local stored = buffer:get(1)

	expectEqual(stored.observation[1], 0.1)
	expectEqual(stored.action[1], 0.5)
end)

--------------------------------------------------
-- Test 9: Full capacity
--------------------------------------------------

test("Buffer detects when it is full", function()
	local buffer = RolloutBuffer.new(2)

	buffer:add(makeExperience(1))
	expect(not buffer:isFull())

	buffer:add(makeExperience(2))
	expect(buffer:isFull())
	expectEqual(buffer:getSize(), 2)
end)

--------------------------------------------------
-- Test 10: Reject additions when full
--------------------------------------------------

test("Full buffer rejects additional experiences", function()
	local buffer = RolloutBuffer.new(1)

	buffer:add(makeExperience(1))

	expectError(function()
		buffer:add(makeExperience(2))
	end)

	expectEqual(buffer:getSize(), 1)
	expectEqual(buffer:get(1).reward, 1)
end)

--------------------------------------------------
-- Test 11: Clear resets the buffer
--------------------------------------------------

test("Clear removes experiences and resets state", function()
	local buffer = RolloutBuffer.new(3)

	buffer:add(makeExperience(1))
	buffer:add(makeExperience(2))
	buffer:clear()

	expectEqual(buffer:getSize(), 0)
	expect(not buffer:isFull())

	buffer:add(makeExperience(3))

	expectEqual(buffer:getSize(), 1)
	expectEqual(buffer:get(1).reward, 3)
end)

--------------------------------------------------
-- Test 12: Get validates index
--------------------------------------------------

test("Get rejects invalid indices", function()
	local buffer = RolloutBuffer.new(3)

	buffer:add(makeExperience())

	expectError(function()
		buffer:get(0)
	end)

	expectError(function()
		buffer:get(2)
	end)

	expectError(function()
		buffer:get(1.5)
	end)
end)

--------------------------------------------------
-- Test 13: Reject inconsistent dimensions
--------------------------------------------------

test("Add rejects mismatched observation or action sizes", function()
	local buffer = RolloutBuffer.new(5)

	buffer:add(makeExperience())

	expectError(function()
		buffer:add(makeExperience(
			1,
			{1, 2},
			{0.5, -0.5}
		))
	end)

	expectError(function()
		buffer:add(makeExperience(
			1,
			{1, 2, 3},
			{0.5}
		))
	end)

	expectEqual(buffer:getSize(), 1)
end)

--------------------------------------------------
-- Test 14: Reject invalid numeric values
--------------------------------------------------

test("Add rejects NaN and infinite values", function()
	local buffer = RolloutBuffer.new(5)

	local badReward = makeExperience(1)
	badReward.reward = 0 / 0

	expectError(function()
		buffer:add(badReward)
	end)

	local badObservation = makeExperience(1)
	badObservation.observation[1] = math.huge

	expectError(function()
		buffer:add(badObservation)
	end)

	local badValue = makeExperience(1)
	badValue.value = -math.huge

	expectError(function()
		buffer:add(badValue)
	end)

	expectEqual(buffer:getSize(), 0)
end)

--------------------------------------------------
-- Test 15: Preserve episode flags and getAll
--------------------------------------------------

test("GetAll preserves experiences and episode flags", function()
	local buffer = RolloutBuffer.new(5)
	local experience = makeExperience(1)

	experience.terminated = true
	experience.truncated = false
	experience.nextValue = 0

	buffer:add(experience)

	local second = makeExperience(2)
	second.terminated = false
	second.truncated = true
	second.nextValue = 0.7

	buffer:add(second)

	local all = buffer:getAll()

	expectEqual(#all, 2)

	expectEqual(all[1].terminated, true)
	expectEqual(all[1].truncated, false)
	expectEqual(all[1].nextValue, 0)

	expectEqual(all[2].terminated, false)
	expectEqual(all[2].truncated, true)
	expectEqual(all[2].nextValue, 0.7)

	-- getAll must also return independent vector copies.
	all[1].observation[1] = 999

	expectEqual(buffer:get(1).observation[1], 0.1)
end)

--------------------------------------------------
-- Results
--------------------------------------------------

print("--------------------------------")
print("RolloutBuffer Test Results")
print("Passed: " .. passed .. "/15")
print("Failed: " .. failed .. "/15")
print("--------------------------------")

if failed > 0 then
	error(
		string.format(
			"RolloutBuffer tests failed: %d passed, %d failed",
			passed,
			failed
		)
	)
end

print("All RolloutBuffer tests passed!")
