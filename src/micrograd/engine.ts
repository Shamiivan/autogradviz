export type ValueInput = Value | number;

class Value {
  data: number = 0;
  grad: number = 0;
  private _prev: Set<Value> = new Set();
  private _backward: () => void = () => { };
  private _op: string = "";

  constructor(
    data: number, children: Iterable<Value> = [], op = ""
  ) {
    this.data = data;
    this.grad = 0;
    this._prev = new Set(children);
    this._op = op;
  }
  // Set the backward function
  setBackward(fn: () => void) {
    this._backward = fn;
  }

  backward() {
    const topo: Value[] = [];
    const visited = new Set<Value>();

    const buildTopo = (node: Value) => {
      if (visited.has(node)) return;

      visited.add(node);
      node._prev.forEach(child => buildTopo(child));
      topo.push(node);
    };
    buildTopo(this);
    this.grad = 1;

    for (let i = topo.length - 1; i >= 0; i--) {
      topo[i]._backward();
    }
  }

  toString() {
    return `Value(data=${this.data}, grad=${this.grad})`;
  }
  get op() {
    return this._op;
  }
  get prev() {
    return this._prev;
  }
}

function toValue(x: ValueInput): Value {
  return x instanceof Value ? x : new Value(x);
}

// basic operations
export function add(a: ValueInput, b: ValueInput): Value {
  a = toValue(a);
  b = toValue(b);
  const out = new Value(a.data + b.data, [a, b], "+");

  out.setBackward(() => {
    a.grad += out.grad;
    b.grad += out.grad;
  });
  return out;
}
export function sub(a: ValueInput, b: ValueInput): Value { return add(a, neg(b)); }

export function mul(a: ValueInput, b: ValueInput): Value {
  a = toValue(a);
  b = toValue(b);
  const out = new Value(a.data * b.data, [a, b], "*");

  out.setBackward(() => {
    a.grad += b.data * out.grad;
    b.grad += a.data * out.grad;
  });
  return out;
}
export function div(a: ValueInput, b: ValueInput): Value { return mul(a, pow(b, -1)); }

export function pow(a: ValueInput, exponent: number): Value {
  a = toValue(a);
  const out = new Value(Math.pow(a.data, exponent), [a], `**${exponent}`);

  out.setBackward(() => {
    a.grad += exponent * Math.pow(a.data, exponent - 1) * out.grad;
  });
  return out;
}

export function neg(a: ValueInput): Value {
  a = toValue(a);
  const out = new Value(-a.data, [a], "neg");

  out.setBackward(() => {
    a.grad += -1 * out.grad;
  });
  return out;
}
export function exp(a: ValueInput): Value {
  a = toValue(a);
  const out = new Value(Math.exp(a.data), [a], "exp");

  out.setBackward(() => {
    a.grad += out.data * out.grad;
  });
  return out;
}

// activation functions

export function relu(a: ValueInput): Value {
  a = toValue(a);
  const out = new Value(a.data < 0 ? 0 : a.data, [a], "ReLU");

  out.setBackward(() => {
    a.grad += (out.data > 0 ? 1 : 0) * out.grad;
  });
  return out;
}

export function sigmoid(a: ValueInput): Value {
  a = toValue(a);
  const s = 1 / (1 + Math.exp(-a.data));
  const out = new Value(s, [a], "sigmoid");

  out.setBackward(() => {
    a.grad += s * (1 - s) * out.grad;
  });
  return out;
}
export function tanh(a: ValueInput): Value {
  a = toValue(a);
  const t = Math.tanh(a.data);
  const out = new Value(t, [a], "tanh");

  out.setBackward(() => {
    a.grad += (1 - t * t) * out.grad;
  });
  return out;
}
export { Value, toValue };