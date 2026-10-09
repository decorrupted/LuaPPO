--!strict

--[[
	RolloutBuffer.lua

	Stores experiences collected by an RL agent.

	Each experience contains:
		- observation: state observed before acting
		- action: action selected by the policy
		- reward: reward received
		- value: critic's value estimate
		- logProb: log probability of the selected action
		- terminated: episode ended naturally
		- truncated: episode ended due to a time limit
		- nextValue: critic estimate of the next state

	Designed for PPO and GAE.

	Observations and actions are copied when inserted
	so later modifications won't corrupt stored data.
]]

local RolloutBuffer = {}
RolloutBuffer.__index = RolloutBuffer

--------------------------------------------------
-- Types
--------------------------------------------------

export type Experience = {
	observation: {number},
	action: {number},
	reward: number,
	value: number,
	logProb: number,

	terminated: boolean,
	truncated: boolean,

	nextValue: number,
}

export type RolloutBufferType = {
	capacity: number,
	size: number,
	position: number,

	observations: {{number}},
	actions: {{number}},
	rewards: {number},
	values: {number},
	logProbs: {number},

	terminated: {boolean},
	truncated: {boolean},
	nextValues: {number},

	add: (
		self: RolloutBufferType,
		experience: Experience
	) -> (),

	get: (
		self: RolloutBufferType,
		index: number
	) -> Experience,

	getAll: (
		self: RolloutBufferType
	) -> {Experience},

	clear: (self: RolloutBufferType) -> (),
	isFull: (self: RolloutBufferType) -> boolean,
	getSize: (self: RolloutBufferType) -> number,
}

--------------------------------------------------
-- Helpers
--------------------------------------------------

local function copyVector(vector: {number}): {number}
	local result = table.create(#vector)

	for i = 1, #vector do
		result[i] = vector[i]
	end

	return result
end

local function validateVector(
	vector: {number},
	name: string
)
	assert(
		type(vector) == "table",
		name .. " must be a vector/table"
	)

	for i = 1, #vector do
		assert(
			type(vector[i]) == "number"
				and vector[i] == vector[i]
				and vector[i] < math.huge
				and vector[i] > -math.huge,
			string.format(
				"%s[%d] must be a finite number",
				name,
				i
			)
		)
	end
end

local function validateFiniteNumber(
	value: number,
	name: string
)
	assert(
		type(value) == "number"
			and value == value
			and value < math.huge
			and value > -math.huge,
		name .. " must be a finite number"
	)
end

--------------------------------------------------
-- Constructor
--------------------------------------------------

function RolloutBuffer.new(
	capacity: number
): RolloutBufferType

	assert(
		type(capacity) == "number"
			and capacity == capacity
			and capacity < math.huge
			and capacity % 1 == 0
			and capacity >= 1,
		"RolloutBuffer.new: capacity must be a positive integer"
	)

	local self: RolloutBufferType = setmetatable({
		capacity = capacity,
		size = 0,
		position = 1,

		observations = table.create(capacity),
		actions = table.create(capacity),
		rewards = table.create(capacity),
		values = table.create(capacity),
		logProbs = table.create(capacity),

		terminated = table.create(capacity),
		truncated = table.create(capacity),
		nextValues = table.create(capacity),
	}, RolloutBuffer)

	return self
end

--------------------------------------------------
-- Add experience
--------------------------------------------------

function RolloutBuffer:add(
	experience: Experience
)
	assert(
		type(experience) == "table",
		"RolloutBuffer.add: experience must be a table"
	)

	assert(
		self.size < self.capacity,
		"RolloutBuffer.add: buffer is full; process or clear it before adding more"
	)

	validateVector(
		experience.observation,
		"observation"
	)

	validateVector(
		experience.action,
		"action"
	)

	assert(
		#experience.observation > 0,
		"observation must not be empty"
	)

	assert(
		#experience.action > 0,
		"action must not be empty"
	)

	validateFiniteNumber(experience.reward, "reward")
	validateFiniteNumber(experience.value, "value")
	validateFiniteNumber(experience.logProb, "logProb")
	validateFiniteNumber(experience.nextValue, "nextValue")

	assert(
		type(experience.terminated) == "boolean",
		"terminated must be a boolean"
	)

	assert(
		type(experience.truncated) == "boolean",
		"truncated must be a boolean"
	)

	-- Keep observation and action dimensions consistent
	-- throughout a rollout.
	if self.size > 0 then
		assert(
			#experience.observation == #self.observations[1],
			"observation dimension mismatch"
		)

		assert(
			#experience.action == #self.actions[1],
			"action dimension mismatch"
		)
	end

	local index = self.position

	self.observations[index] =
		copyVector(experience.observation)

	self.actions[index] =
		copyVector(experience.action)

	self.rewards[index] = experience.reward
	self.values[index] = experience.value
	self.logProbs[index] = experience.logProb

	self.terminated[index] = experience.terminated
	self.truncated[index] = experience.truncated

	self.nextValues[index] = experience.nextValue

	self.size += 1
	self.position += 1
end

--------------------------------------------------
-- Get one experience
--------------------------------------------------

function RolloutBuffer:get(
	index: number
): Experience

	assert(
		type(index) == "number"
			and index % 1 == 0
			and index >= 1
			and index <= self.size,
		"RolloutBuffer.get: index out of range"
	)

	return {
		observation = copyVector(self.observations[index]),
		action = copyVector(self.actions[index]),

		reward = self.rewards[index],
		value = self.values[index],
		logProb = self.logProbs[index],

		terminated = self.terminated[index],
		truncated = self.truncated[index],

		nextValue = self.nextValues[index],
	}
end

--------------------------------------------------
-- Get all experiences
--------------------------------------------------

function RolloutBuffer:getAll(): {Experience}
	local result = table.create(self.size)

	for i = 1, self.size do
		result[i] = self:get(i)
	end

	return result
end

--------------------------------------------------
-- Capacity and size
--------------------------------------------------

function RolloutBuffer:isFull(): boolean
	return self.size >= self.capacity
end

function RolloutBuffer:getSize(): number
	return self.size
end

--------------------------------------------------
-- Clear buffer
--------------------------------------------------

function RolloutBuffer:clear()
	table.clear(self.observations)
	table.clear(self.actions)
	table.clear(self.rewards)
	table.clear(self.values)
	table.clear(self.logProbs)

	table.clear(self.terminated)
	table.clear(self.truncated)
	table.clear(self.nextValues)

	self.size = 0
	self.position = 1
end

return RolloutBuffer
