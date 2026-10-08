--!strict

--[[

    Matrix - Lightweight matrix/vector library for LuaPPO

]]
local Matrix = {};

export type Vector = { number };
export type Mat = {
    data: { { number } },
    rows: number,
    cols: number,
};

local RNG = Random.new();

-- Consturctors --
function Matrix.vector(size: number, value: number?): Vector
    local v = table.create(size);
    local x = value or 0;
    for i = 1, size do
        v[i] = x;
    end;

    return v;
end;

function Matrix.zeros(rows: number, cols: number): Mat
    local data = table.create(rows);

    for i = 1, rows do
        local row = table.create(cols, 0);
        data[i] = row;
    end;

    return {
        data = data,
        rows = rows,
        cols = cols,
    };
end;

function Matrix.ones(rows: number, cols: number): Mat
    local data = table.create(rows);

    for i = 1, rows do
        local row = table.create(cols);

        for j = 1, cols do
            row[j] = 1;
        end;

        data[i] = row;
    end;

    return {
        data = data,
        rows = rows,
        cols = cols,
    };
end;

function Matrix.fromArray(data: { { number } }): Mat
    return {
        data = data,
        rows = #data,
        cols = #data > 0 and #data[1] or 0,
    };
end;

function Matrix.xavier(rows: number, cols: number): Mat
    local limit = math.sqrt(6 / (rows + cols));

    local data = table.create(rows);

    for i = 1, rows do
        local row = table.create(cols);
        for j = 1, cols do
            row[j] = RNG:NextNumber(-limit, limit);
        end;

        data[i] = row;
    end;

    return {
        data = data,
        rows = rows,
        cols = cols,
    };
end;

function Matrix.he(rows: number, cols: number): Mat
    local std = math.sqrt(2 / cols);
    local data = table.create(rows);

    for i = 1, rows do
        local row = table.create(cols);

        for j = 1, cols do
            local u1 = math.max(RNG:NextNumber(), 1e-7);
            local u2 = RNG:NextNumber();

            local z = math.sqrt(-2 * math.log(u1)) * math.cos(2 * math.pi * u2);
            row[j] = z * std;
        end;

        data[i] = row;
    end;

    return {
        data = data,
        cols = cols,
        rows = rows,
    };
end;

function Matrix.copy(a: Mat): Mat
    local result = table.create(a.rows);

    for i = 1, a.rows do
        local row = table.create(a.cols);

        for j = 1, a.cols do
            row[j] = a.data[i][j];
        end;

        result[i] = row;
    end;

    return {
        data = result,
        rows = a.rows,
        cols = a.cols,
    };
end;

function Matrix.copyVector(v: Vector): Vector
    local result = table.create(#v);

    for i = 1, #v do
        result[i] = v[i];
    end;

    return result;
end;

function Matrix.add(a: Mat, b: Mat): Mat
    assert(a.rows == b.rows and a.cols == b.cols, "Matrix dimensions do not match");
    local result = table.create(a.rows);
    for i = 1, a.rows do
        local row = table.create(a.cols);
        local ar = a.data[i];
        local br = b.data[i];
        for j = 1, a.cols do
            row[j] = ar[j] + br[j];
        end;
        result[i] = row;
    end;
    return {
        data = result,
        rows = a.rows,
        cols = a.cols,
    };
end;

function Matrix.sub(a: Mat, b: Mat): Mat
    assert(a.rows == b.rows and a.cols == b.cols, "Matrix dimensions do not match");
    local result = table.create(a.rows);
    for i = 1, a.rows do
        local row = table.create(a.cols);
        local ar = a.data[i];
        local br = b.data[i];
        for j = 1, a.cols do
            row[j] = ar[j] - br[j];
        end;
        result[i] = row;
    end;
    return { data = result, rows = a.rows, cols = a.cols };
end;

function Matrix.mulScalar(a: Mat, scalar: number): Mat
    local result = table.create(a.rows);
    for i = 1, a.rows do
        local row = table.create(a.cols);
        local ar = a.data[i];
        for j = 1, a.cols do
            row[j] = ar[j] * scalar;
        end;
        result[i] = row;
    end;
    return { data = result, rows = a.rows, cols = a.cols };
end;

function Matrix.matmul(a: Mat, b: Mat): Mat
    assert(a.cols == b.rows, "Cannot multiply matrices: A.cols must equal B.rows");
    local result = table.create(a.rows);
    for i = 1, a.rows do
        local row = table.create(b.cols, 0);
        local aRow = a.data[i];
        for k = 1, a.cols do
            local aik = aRow[k];
            local bRow = b.data[k];
            for j = 1, b.cols do
                row[j] += aik * bRow[j];
            end;
        end;
        result[i] = row;
    end;
    return { data = result, rows = a.rows, cols = b.cols };
end;

function Matrix.matVec(a: Mat, v: Vector): Vector
    assert(a.cols == #v, "Matrix columns must equal vector size");
    local result = table.create(a.rows);
    for i = 1, a.rows do
        local row = a.data[i];
        local sum = 0;
        for j = 1, a.cols do
            sum += row[j] * v[j];
        end;
        result[i] = sum;
    end;
    return result;
end;

function Matrix.transpose(a: Mat): Mat
    local result = table.create(a.cols);
    for j = 1, a.cols do
        local row = table.create(a.rows);
        for i = 1, a.rows do
            row[i] = a.data[i][j];
        end;
        result[j] = row;
    end;
    return { data = result, rows = a.cols, cols = a.rows };
end;

function Matrix.vectorAdd(a: Vector, b: Vector): Vector
    assert(#a == #b, "Vector dimensions do not match");
    local result = table.create(#a);
    for i = 1, #a do
        result[i] = a[i] + b[i];
    end;
    return result;
end;

function Matrix.vectorSub(a: Vector, b: Vector): Vector
    assert(#a == #b, "Vector dimensions do not match");
    local result = table.create(#a);
    for i = 1, #a do
        result[i] = a[i] - b[i];
    end;
    return result;
end;

function Matrix.vectorMul(a: Vector, scalar: number): Vector
    local result = table.create(#a);
    for i = 1, #a do
        result[i] = a[i] * scalar;
    end;
    return result;
end;

function Matrix.dot(a: Vector, b: Vector): number
    assert(#a == #b, "Vector dimensions do not match");
    local sum = 0;
    for i = 1, #a do
        sum += a[i] * b[i];
    end;
    return sum;
end;

function Matrix.sum(a: Vector): number
    local total = 0;
    for i = 1, #a do
        total += a[i];
    end;
    return total;
end;

function Matrix.map(a: Vector, fn: (number) -> number): Vector
    local result = table.create(#a);
    for i = 1, #a do
        result[i] = fn(a[i]);
    end;
    return result;
end;

function Matrix.relu(a: Vector): Vector
    local result = table.create(#a);
    for i = 1, #a do
        local x = a[i];
        if x > 0 then
            result[i] = x;
        else
            result[i] = 0;
        end;
    end;
    return result;
end;

function Matrix.reluDerivative(a: Vector): Vector
    local result = table.create(#a);
    for i = 1, #a do
        if a[i] > 0 then
            result[i] = 1;
        else
            result[i] = 0;
        end;
    end;
    return result;
end;

function Matrix.vectorNorm(a: Vector): number
    local sum = 0;
    for i = 1, #a do
        sum += a[i] * a[i];
    end;
    return math.sqrt(sum);
end;

function Matrix.clipVector(a: Vector, minValue: number, maxValue: number): Vector
    local result = table.create(#a);
    for i = 1, #a do
        result[i] = math.clamp(a[i], minValue, maxValue);
    end;
    return result;
end;

function Matrix.print(a: Mat)
    for i = 1, a.rows do
        print(table.concat(a.data[i], "\t"));
    end;
end;

function Matrix.printVector(v: Vector)
    print(table.concat(v, "\t"));
end;

return Matrix;
