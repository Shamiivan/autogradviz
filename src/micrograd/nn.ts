import { Value, add, mul, sigmoid } from "./engine";

// base class
abstract class Module {
  abstract parameters(): Value[];
  zeroGrad() {
    for (const p of this.parameters()) {
      p.grad = 0;
    }
  }
}


class Neuron extends Module {
  w: Value[]
  b: Value;
  isLinear: boolean;

  constructor(numberOfInputs: number, isLinear = false) {
    super();
    this.w = Array.from({ length: numberOfInputs }, () => new Value(Math.random() * 2 - 1));
    this.b = new Value(0);
    this.isLinear = isLinear;
  }
  forward(x: Value[]): Value {
    let act = this.b;

    for (let i = 0; i < this.w.length; i++) {
      act = add(act, mul(this.w[i], x[i]));
    }
    return this.isLinear ? act : sigmoid(act);
  }


  parameters(): Value[] {
    return [...this.w, this.b];
  }
  toString() {
    return `Weights: [${this.w.map(w => w.toString()).join(", ")}], \nBias: ${this.b.toString()} \nLinear: ${this.isLinear}`;
  }
}

class Layer extends Module {
  neurons: Neuron[];

  constructor(numberOfInputs: number, numberOfNeurons: number, isLinear = false) {
    super();
    this.neurons = Array.from({ length: numberOfNeurons }, () => new Neuron(numberOfInputs, isLinear));
  }

  forward(x: Value[]): Value | Value[] {
    const out = this.neurons.map(neuron => neuron.forward(x));
    return out.length === 1 ? out[0] : out;
  }

  parameters(): Value[] {
    return this.neurons.flatMap(neuron => neuron.parameters());
  }

  toString() {
    return `Layer: \n${this.neurons.map(neuron => neuron.toString()).join("\t")}`;
  }
}
class MLP extends Module {
  layers: Layer[];

  constructor(numberOfInputs: number, layerSizes: number[]) {
    super();
    const sizes = [numberOfInputs, ...layerSizes];
    this.layers = [];
    for (let i = 0; i < layerSizes.length; i++) {
      const isLinear = i === layerSizes.length - 1;
      this.layers.push(new Layer(sizes[i], sizes[i + 1], isLinear));
    }
  }

  forward(x: Value[]): Value | Value[] {
    let out: Value | Value[] = x;
    for (const layer of this.layers) {
      out = layer.forward(Array.isArray(out) ? out : [out]);
    }
    return out;
  }
  parameters(): Value[] {
    return this.layers.flatMap(layer => layer.parameters());
  }
  toString() {
    return `MLP: \n${this.layers.map((layer, idx) => ` Layer ${idx}:\n${layer.toString()}`).join("\n")}`;
  }
}

export { Module, Neuron, Layer, MLP };