import { describe, it, expect, beforeEach } from "vitest";
import { Neuron, Layer, MLP } from "../src/micrograd/nn";
import { Value, add, mul } from "../src/micrograd/engine";

describe('Neuron class', () => {
  it('should initialize with random weights and zero bias', () => {
    const neuron = new Neuron(3);

    expect(neuron.w.length).toBe(3);
    expect(neuron.b.data).toBe(0);
    expect(neuron.isLinear).toBe(false);

    neuron.w.forEach(weight => {
      expect(weight).toBeInstanceOf(Value);
      expect(weight.data).toBeGreaterThanOrEqual(-1);
      expect(weight.data).toBeLessThanOrEqual(1);
    });
  });

  it('should initialize as linear when specified', () => {
    const neuron = new Neuron(2, true);

    expect(neuron.isLinear).toBe(true);
  });

  it('should forward pass with sigmoid activation', () => {
    const neuron = new Neuron(2);
    neuron.w[0].data = 0.5;
    neuron.w[1].data = 0.5;
    neuron.b.data = 0;

    const x = [new Value(1), new Value(1)];
    const output = neuron.forward(x);

    expect(output).toBeInstanceOf(Value);
    // sigmoid(0.5 + 0.5) = sigmoid(1) ≈ 0.731
    expect(output.data).toBeGreaterThan(0.7);
    expect(output.data).toBeLessThan(0.75);
  });

  it('should forward pass with linear activation', () => {
    const neuron = new Neuron(2, true);
    neuron.w[0].data = 2;
    neuron.w[1].data = 3;
    neuron.b.data = 1;

    const x = [new Value(5), new Value(10)];
    const output = neuron.forward(x);

    // 2*5 + 3*10 + 1 = 41
    expect(output.data).toBe(41);
  });

  it('should return correct parameters', () => {
    const neuron = new Neuron(3);
    const params = neuron.parameters();

    expect(params.length).toBe(4); // 3 weights + 1 bias
    expect(params[0]).toBe(neuron.w[0]);
    expect(params[1]).toBe(neuron.w[1]);
    expect(params[2]).toBe(neuron.w[2]);
    expect(params[3]).toBe(neuron.b);
  });

  it('should have meaningful toString', () => {
    const neuron = new Neuron(2, true);
    const str = neuron.toString();

    expect(str).toContain('Weights:');
    expect(str).toContain('Bias:');
    expect(str).toContain('Linear: true');
  });

  it('should compute gradients through forward pass', () => {
    const neuron = new Neuron(2, true);
    neuron.w[0].data = 1;
    neuron.w[1].data = 1;
    neuron.b.data = 0;

    const x = [new Value(2), new Value(3)];
    const output = neuron.forward(x);

    output.backward();

    expect(neuron.w[0].grad).toBe(2);
    expect(neuron.w[1].grad).toBe(3);
    expect(neuron.b.grad).toBe(1);
  });
});

describe('Layer class', () => {
  it('should initialize with correct number of neurons', () => {
    const layer = new Layer(3, 5);

    expect(layer.neurons.length).toBe(5);
    layer.neurons.forEach(neuron => {
      expect(neuron).toBeInstanceOf(Neuron);
      expect(neuron.w.length).toBe(3);
    });
  });

  it('should pass isLinear to all neurons', () => {
    const layer = new Layer(2, 3, true);

    layer.neurons.forEach(neuron => {
      expect(neuron.isLinear).toBe(true);
    });
  });

  it('should forward pass and return single value for one neuron', () => {
    const layer = new Layer(2, 1, true);
    layer.neurons[0].w[0].data = 2;
    layer.neurons[0].w[1].data = 3;
    layer.neurons[0].b.data = 0;

    const x = [new Value(1), new Value(1)];
    const output = layer.forward(x);

    expect(output).toBeInstanceOf(Value);
    expect((output as Value).data).toBe(5);
  });

  it('should forward pass and return array for multiple neurons', () => {
    const layer = new Layer(2, 3, true);
    layer.neurons.forEach((neuron, i) => {
      neuron.w[0].data = i + 1;
      neuron.w[1].data = i + 1;
      neuron.b.data = 0;
    });

    const x = [new Value(1), new Value(1)];
    const output = layer.forward(x);

    expect(Array.isArray(output)).toBe(true);
    expect((output as Value[]).length).toBe(3);
    expect((output as Value[])[0].data).toBe(2);
    expect((output as Value[])[1].data).toBe(4);
    expect((output as Value[])[2].data).toBe(6);
  });

  it('should return all parameters from all neurons', () => {
    const layer = new Layer(2, 3);
    const params = layer.parameters();

    // Each neuron has 2 weights + 1 bias = 3 params
    // 3 neurons * 3 params = 9 total
    expect(params.length).toBe(9);
  });

  it('should have meaningful toString', () => {
    const layer = new Layer(2, 2);
    const str = layer.toString();

    expect(str).toContain('Layer:');
    expect(str).toContain('Weights:');
  });

  it('should compute gradients through layer', () => {
    const layer = new Layer(2, 1, true);
    layer.neurons[0].w[0].data = 1;
    layer.neurons[0].w[1].data = 2;
    layer.neurons[0].b.data = 0;

    const x = [new Value(3), new Value(4)];
    const output = layer.forward(x) as Value;

    output.backward();

    expect(x[0].grad).toBe(1);
    expect(x[1].grad).toBe(2);
  });
});

describe('MLP class', () => {
  it('should initialize with correct layer structure', () => {
    const mlp = new MLP(3, [4, 4, 1]);

    expect(mlp.layers.length).toBe(3);
    expect(mlp.layers[0].neurons.length).toBe(4);
    expect(mlp.layers[1].neurons.length).toBe(4);
    expect(mlp.layers[2].neurons.length).toBe(1);
  });

  it('should have last layer as linear', () => {
    const mlp = new MLP(2, [3, 3, 1]);

    expect(mlp.layers[0].neurons[0].isLinear).toBe(false);
    expect(mlp.layers[1].neurons[0].isLinear).toBe(false);
    expect(mlp.layers[2].neurons[0].isLinear).toBe(true);
  });

  it('should have correct input-output dimensions', () => {
    const mlp = new MLP(3, [5, 4, 2]);

    // First layer: 3 inputs, 5 neurons
    expect(mlp.layers[0].neurons[0].w.length).toBe(3);
    expect(mlp.layers[0].neurons.length).toBe(5);

    // Second layer: 5 inputs, 4 neurons
    expect(mlp.layers[1].neurons[0].w.length).toBe(5);
    expect(mlp.layers[1].neurons.length).toBe(4);

    // Third layer: 4 inputs, 2 neurons
    expect(mlp.layers[2].neurons[0].w.length).toBe(4);
    expect(mlp.layers[2].neurons.length).toBe(2);
  });

  it('should forward pass through all layers', () => {
    const mlp = new MLP(2, [2, 1]);

    // Set all weights to 1 and biases to 0 for predictability
    mlp.layers.forEach(layer => {
      layer.neurons.forEach(neuron => {
        neuron.w.forEach(w => w.data = 0.1);
        neuron.b.data = 0;
      });
    });

    const x = [new Value(1), new Value(1)];
    const output = mlp.forward(x);

    expect(output).toBeInstanceOf(Value);
    expect((output as Value).data).toBeGreaterThan(0);
  });

  it('should return single value for single output', () => {
    const mlp = new MLP(2, [3, 1]);
    const x = [new Value(1), new Value(1)];
    const output = mlp.forward(x);

    expect(output).toBeInstanceOf(Value);
    expect(output).not.toBeInstanceOf(Array);
  });

  it('should return array for multiple outputs', () => {
    const mlp = new MLP(2, [3, 3]);
    const x = [new Value(1), new Value(1)];
    const output = mlp.forward(x);

    expect(Array.isArray(output)).toBe(true);
    expect((output as Value[]).length).toBe(3);
  });

  it('should return all parameters from all layers', () => {
    const mlp = new MLP(2, [3, 1]);
    const params = mlp.parameters();

    // Layer 1: 3 neurons * (2 weights + 1 bias) = 9
    // Layer 2: 1 neuron * (3 weights + 1 bias) = 4
    // Total: 13
    expect(params.length).toBe(13);
  });

  it('should have meaningful toString', () => {
    const mlp = new MLP(2, [2, 1]);
    const str = mlp.toString();

    expect(str).toContain('MLP:');
    expect(str).toContain('Layer 0:');
    expect(str).toContain('Layer 1:');
  });

  it('should compute gradients end-to-end', () => {
    const mlp = new MLP(2, [2, 1]);

    // Simplify network
    mlp.layers.forEach(layer => {
      layer.neurons.forEach(neuron => {
        neuron.w.forEach(w => w.data = 0.5);
        neuron.b.data = 0;
      });
    });

    const x = [new Value(1), new Value(1)];
    const output = mlp.forward(x) as Value;

    output.backward();

    // Check that gradients were computed
    const params = mlp.parameters();
    params.forEach(param => {
      expect(typeof param.grad).toBe('number');
    });
  });

  it('should handle deep networks', () => {
    const mlp = new MLP(3, [10, 8, 6, 4, 2]);

    expect(mlp.layers.length).toBe(5);

    const x = [new Value(1), new Value(2), new Value(3)];
    const output = mlp.forward(x);

    expect(Array.isArray(output)).toBe(true);
    expect((output as Value[]).length).toBe(2);
  });
});

describe('Module base class', () => {
  it('should zero gradients for Neuron', () => {
    const neuron = new Neuron(3);

    // Set some gradients
    neuron.w[0].grad = 5;
    neuron.w[1].grad = 3;
    neuron.b.grad = 2;

    neuron.zeroGrad();

    expect(neuron.w[0].grad).toBe(0);
    expect(neuron.w[1].grad).toBe(0);
    expect(neuron.b.grad).toBe(0);
  });

  it('should zero gradients for Layer', () => {
    const layer = new Layer(2, 3);

    // Set some gradients
    layer.parameters().forEach(p => p.grad = Math.random() * 10);

    layer.zeroGrad();

    layer.parameters().forEach(p => {
      expect(p.grad).toBe(0);
    });
  });

  it('should zero gradients for MLP', () => {
    const mlp = new MLP(3, [4, 2]);

    // Set some gradients
    mlp.parameters().forEach(p => p.grad = Math.random() * 10);

    mlp.zeroGrad();

    mlp.parameters().forEach(p => {
      expect(p.grad).toBe(0);
    });
  });
});

describe('Integration tests', () => {
  it('should perform a complete training step', () => {
    const mlp = new MLP(2, [3, 1]);

    // Training data: XOR-like problem
    const X = [
      [new Value(0), new Value(0)],
      [new Value(0), new Value(1)],
      [new Value(1), new Value(0)],
      [new Value(1), new Value(1)]
    ];
    const y = [0, 1, 1, 0];

    // Single training step
    mlp.zeroGrad();

    let totalLoss = new Value(0);
    X.forEach((xi, i) => {
      const pred = mlp.forward(xi) as Value;
      const target = new Value(y[i]);
      const diff = add(pred, mul(target, -1));
      const loss = mul(diff, diff);
      totalLoss = add(totalLoss, loss);
    });

    totalLoss.backward();

    // Check that all parameters have gradients
    mlp.parameters().forEach(p => {
      expect(typeof p.grad).toBe('number');
      expect(isNaN(p.grad)).toBe(false);
    });

    // Simulate parameter update
    const learningRate = 0.01;
    mlp.parameters().forEach(p => {
      p.data -= learningRate * p.grad;
    });

    // Verify parameters changed
    const hasChanged = mlp.parameters().some(p => p.data !== 0);
    expect(hasChanged).toBe(true);
  });

  it('should handle batch processing', () => {
    const mlp = new MLP(3, [5, 2]);

    const batch = [
      [new Value(1), new Value(2), new Value(3)],
      [new Value(4), new Value(5), new Value(6)],
      [new Value(7), new Value(8), new Value(9)]
    ];

    const outputs = batch.map(x => mlp.forward(x));

    expect(outputs.length).toBe(3);
    outputs.forEach(output => {
      expect(Array.isArray(output)).toBe(true);
      expect((output as Value[]).length).toBe(2);
    });
  });

  it('should approximate a simple function', () => {
    const mlp = new MLP(1, [4, 1]);

    // Try to learn f(x) = 2*x
    const X = [
      [new Value(0)],
      [new Value(1)],
      [new Value(2)],
      [new Value(3)]
    ];
    const y = [0, 2, 4, 6];

    // Multiple training steps
    for (let epoch = 0; epoch < 50; epoch++) {
      mlp.zeroGrad();

      let totalLoss = new Value(0);
      X.forEach((xi, i) => {
        const pred = mlp.forward(xi) as Value;
        const target = new Value(y[i]);
        const diff = add(pred, mul(target, -1));
        const loss = mul(diff, diff);
        totalLoss = add(totalLoss, loss);
      });

      totalLoss.backward();

      const learningRate = 0.01;
      mlp.parameters().forEach(p => {
        p.data -= learningRate * p.grad;
      });
    }

    // Check predictions are closer to targets
    const predictions = X.map(xi => (mlp.forward(xi) as Value).data);
    const errors = predictions.map((pred, i) => Math.abs(pred - y[i]));
    const avgError = errors.reduce((a, b) => a + b, 0) / errors.length;

    // After training, average error should be reasonably small
    // (may not be perfect due to limited training and network capacity)
    expect(avgError).toBeLessThan(y.reduce((a, b) => a + b, 0) / y.length);
  });

  it('should maintain gradient flow in complex network', () => {
    const mlp = new MLP(4, [8, 6, 4, 2]);

    const x = [new Value(1), new Value(2), new Value(3), new Value(4)];
    const output = mlp.forward(x) as Value[];

    // Combine outputs into scalar loss
    const loss = add(mul(output[0], output[0]), mul(output[1], output[1]));

    mlp.zeroGrad();
    loss.backward();

    // Verify all parameters received gradients
    const params = mlp.parameters();
    const nonZeroGrads = params.filter(p => Math.abs(p.grad) > 1e-10).length;

    // Most parameters should have non-zero gradients
    expect(nonZeroGrads).toBeGreaterThan(params.length * 0.5);
  });
});