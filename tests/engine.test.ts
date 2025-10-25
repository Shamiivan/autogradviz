import { describe, it, expect } from 'vitest';
import {
  Value,
  add,
  sub,
  mul,
  div,
  pow,
  neg,
  exp,
  relu,
  sigmoid,
  tanh,
  toValue
} from "../src/micrograd/engine";

describe('Value class', () => {
  it('should create a Value with data', () => {
    const v = new Value(5);
    expect(v.data).toBe(5);
    expect(v.grad).toBe(0);
  });

  it('should initialize gradient to 0', () => {
    const v = new Value(10);
    expect(v.grad).toBe(0);
  });

  it('should store operation and previous values', () => {
    const a = new Value(2);
    const b = new Value(3);
    const c = new Value(5, [a, b], '+');

    expect(c.op).toBe('+');
    expect(c.prev.size).toBe(2);
    expect(c.prev.has(a)).toBe(true);
    expect(c.prev.has(b)).toBe(true);
  });

  it('should have a meaningful toString', () => {
    const v = new Value(2.5);
    v.grad = 1.5;
    expect(v.toString()).toBe('Value(data=2.5, grad=1.5)');
  });
});

describe('toValue helper', () => {
  it('should convert number to Value', () => {
    const v = toValue(5);
    expect(v).toBeInstanceOf(Value);
    expect(v.data).toBe(5);
  });

  it('should return Value as-is', () => {
    const original = new Value(10);
    const result = toValue(original);
    expect(result).toBe(original);
  });
});

describe('Arithmetic operations', () => {
  describe('add', () => {
    it('should add two Values', () => {
      const a = new Value(2);
      const b = new Value(3);
      const c = add(a, b);

      expect(c.data).toBe(5);
      expect(c.op).toBe('+');
    });

    it('should add Value and number', () => {
      const a = new Value(2);
      const c = add(a, 3);

      expect(c.data).toBe(5);
    });

    it('should compute gradients correctly', () => {
      const a = new Value(2);
      const b = new Value(3);
      const c = add(a, b);

      c.backward();

      expect(a.grad).toBe(1);
      expect(b.grad).toBe(1);
      expect(c.grad).toBe(1);
    });
  });

  describe('sub', () => {
    it('should subtract two Values', () => {
      const a = new Value(5);
      const b = new Value(3);
      const c = sub(a, b);

      expect(c.data).toBe(2);
    });

    it('should compute gradients correctly', () => {
      const a = new Value(5);
      const b = new Value(3);
      const c = sub(a, b);

      c.backward();

      expect(a.grad).toBe(1);
      expect(b.grad).toBe(-1);
    });
  });

  describe('mul', () => {
    it('should multiply two Values', () => {
      const a = new Value(2);
      const b = new Value(3);
      const c = mul(a, b);

      expect(c.data).toBe(6);
      expect(c.op).toBe('*');
    });

    it('should multiply Value and number', () => {
      const a = new Value(2);
      const c = mul(a, 3);

      expect(c.data).toBe(6);
    });

    it('should compute gradients correctly', () => {
      const a = new Value(2);
      const b = new Value(3);
      const c = mul(a, b);

      c.backward();

      expect(a.grad).toBe(3);
      expect(b.grad).toBe(2);
    });
  });

  describe('div', () => {
    it('should divide two Values', () => {
      const a = new Value(6);
      const b = new Value(3);
      const c = div(a, b);

      expect(c.data).toBe(2);
    });

    it('should compute gradients correctly', () => {
      const a = new Value(6);
      const b = new Value(3);
      const c = div(a, b);

      c.backward();

      expect(a.grad).toBeCloseTo(1 / 3, 5);
      expect(b.grad).toBeCloseTo(-2 / 3, 5);
    });
  });

  describe('pow', () => {
    it('should raise Value to power', () => {
      const a = new Value(2);
      const c = pow(a, 3);

      expect(c.data).toBe(8);
      expect(c.op).toBe('**3');
    });

    it('should compute gradients correctly', () => {
      const a = new Value(2);
      const c = pow(a, 3);

      c.backward();

      expect(a.grad).toBe(12); // 3 * 2^2 = 12
    });

    it('should handle negative exponents', () => {
      const a = new Value(2);
      const c = pow(a, -1);

      expect(c.data).toBe(0.5);
    });

    it('should handle fractional exponents', () => {
      const a = new Value(4);
      const c = pow(a, 0.5);

      expect(c.data).toBe(2);
    });
  });

  describe('neg', () => {
    it('should negate Value', () => {
      const a = new Value(5);
      const b = neg(a);

      expect(b.data).toBe(-5);
      expect(b.op).toBe('neg');
    });

    it('should compute gradients correctly', () => {
      const a = new Value(5);
      const b = neg(a);

      b.backward();

      expect(a.grad).toBe(-1);
    });
  });

  describe('exp', () => {
    it('should compute exponential', () => {
      const a = new Value(0);
      const b = exp(a);

      expect(b.data).toBe(1);
    });

    it('should compute exp for positive values', () => {
      const a = new Value(1);
      const b = exp(a);

      expect(b.data).toBeCloseTo(Math.E, 5);
    });

    it('should compute gradients correctly', () => {
      const a = new Value(2);
      const b = exp(a);

      b.backward();

      expect(a.grad).toBeCloseTo(Math.exp(2), 5);
    });
  });
});

describe('Activation functions', () => {
  describe('relu', () => {
    it('should return 0 for negative input', () => {
      const a = new Value(-5);
      const b = relu(a);

      expect(b.data).toBe(0);
    });

    it('should return input for positive input', () => {
      const a = new Value(5);
      const b = relu(a);

      expect(b.data).toBe(5);
    });

    it('should return 0 for zero input', () => {
      const a = new Value(0);
      const b = relu(a);

      expect(b.data).toBe(0);
    });

    it('should compute gradients correctly for positive input', () => {
      const a = new Value(5);
      const b = relu(a);

      b.backward();

      expect(a.grad).toBe(1);
    });

    it('should compute gradients correctly for negative input', () => {
      const a = new Value(-5);
      const b = relu(a);

      b.backward();

      expect(a.grad).toBe(0);
    });
  });

  describe('sigmoid', () => {
    it('should return 0.5 for input of 0', () => {
      const a = new Value(0);
      const b = sigmoid(a);

      expect(b.data).toBeCloseTo(0.5, 5);
    });

    it('should return value close to 1 for large positive input', () => {
      const a = new Value(10);
      const b = sigmoid(a);

      expect(b.data).toBeGreaterThan(0.99);
    });

    it('should return value close to 0 for large negative input', () => {
      const a = new Value(-10);
      const b = sigmoid(a);

      expect(b.data).toBeLessThan(0.01);
    });

    it('should compute gradients correctly', () => {
      const a = new Value(0);
      const b = sigmoid(a);

      b.backward();

      // sigmoid'(0) = 0.5 * (1 - 0.5) = 0.25
      expect(a.grad).toBeCloseTo(0.25, 5);
    });

    it('should have maximum gradient at 0', () => {
      const testPoints = [-2, -1, 0, 1, 2];
      const gradients = testPoints.map(x => {
        const a = new Value(x);
        const b = sigmoid(a);
        b.backward();
        return a.grad;
      });

      const maxGrad = Math.max(...gradients);
      expect(gradients[2]).toBe(maxGrad);
    });
  });

  describe('tanh', () => {
    it('should return 0 for input of 0', () => {
      const a = new Value(0);
      const b = tanh(a);

      expect(b.data).toBeCloseTo(0, 5);
    });

    it('should return value close to 1 for large positive input', () => {
      const a = new Value(10);
      const b = tanh(a);

      expect(b.data).toBeGreaterThan(0.99);
    });

    it('should return value close to -1 for large negative input', () => {
      const a = new Value(-10);
      const b = tanh(a);

      expect(b.data).toBeLessThan(-0.99);
    });

    it('should compute gradients correctly', () => {
      const a = new Value(0);
      const b = tanh(a);

      b.backward();

      // tanh'(0) = 1 - 0^2 = 1
      expect(a.grad).toBeCloseTo(1, 5);
    });
  });
});

describe('Backpropagation', () => {
  it('should compute gradients through complex expression', () => {
    const a = new Value(2);
    const b = new Value(3);
    const c = add(a, b); // 5
    const d = mul(c, a); // 10

    d.backward();

    expect(a.grad).toBe(7); // dc/da * dd/dc + dd/da = 1*2 + 5 = 7
    expect(b.grad).toBe(2); // dc/db * dd/dc = 1*2 = 2
  });

  it('should handle multiple uses of same variable', () => {
    const a = new Value(2);
    const b = mul(a, a); // a^2

    b.backward();

    expect(a.grad).toBe(4); // 2*a = 4
  });

  it('should compute gradients for nested operations', () => {
    const x = new Value(3);
    const y = mul(x, 2); // 6
    const z = pow(y, 2); // 36

    z.backward();

    // dz/dx = dz/dy * dy/dx = (2*y) * 2 = 2*6*2 = 24
    expect(x.grad).toBe(24);
  });

  it('should handle division in backprop', () => {
    const a = new Value(10);
    const b = new Value(2);
    const c = div(a, b);

    c.backward();

    expect(a.grad).toBeCloseTo(0.5, 5);
    expect(b.grad).toBeCloseTo(-2.5, 5);
  });

  it('should work with activation functions', () => {
    const x = new Value(0);
    const y = sigmoid(x);
    const z = mul(y, 2);

    z.backward();

    // dz/dx = dz/dy * dy/dx = 2 * sigmoid'(0) = 2 * 0.25 = 0.5
    expect(x.grad).toBeCloseTo(0.5, 5);
  });

  it('should accumulate gradients correctly', () => {
    const x = new Value(2);
    const a = mul(x, 3);
    const b = mul(x, 4);
    const c = add(a, b);

    c.backward();

    // dc/dx = da/dx + db/dx = 3 + 4 = 7
    expect(x.grad).toBe(7);
  });

  it('should handle zero gradients', () => {
    const x = new Value(5);
    const y = relu(new Value(-1));
    const z = mul(x, y);

    z.backward();

    expect(x.grad).toBe(0);
  });
});

describe('Complex computational graphs', () => {
  it('should compute gradients for f(x,y) = x*y + sin-like pattern', () => {
    const x = new Value(2);
    const y = new Value(3);
    const a = mul(x, y); // 6
    const b = add(x, y); // 5
    const c = mul(a, b); // 30

    c.backward();

    expect(c.data).toBe(30); // (2*3) * (2+3) = 6 * 5 = 30
    expect(x.grad).toBe(21); // y*b + a = 3*5 + 6 = 21
    expect(y.grad).toBe(16); // x*b + a = 2*5 + 6 = 16
  });

  it('should handle deep chains', () => {
    const x = new Value(2);
    let result = x;

    for (let i = 0; i < 5; i++) {
      result = mul(result, 2);
    }

    result.backward();

    expect(result.data).toBe(64); // 2^6
    expect(x.grad).toBe(32); // Product rule chain
  });
});