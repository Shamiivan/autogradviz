import { Value } from "./engine";

// Processed sample ready for training
export interface IrisSample {
  inputs: Value[];   // [4 features: sepal length, sepal width, petal length, petal width]
  outputs: Value[];  // [3 one-hot encoded: setosa, versicolor, virginica]
  label: string;     // Original label for display
}

// Map species to one-hot encoding
const speciesMap: { [key: string]: number[] } = {
  "setosa": [1, 0, 0],
  "versicolor": [0, 1, 0],
  "virginica": [0, 0, 1],
  // Also support full names in case CSV format varies
  "Iris-setosa": [1, 0, 0],
  "Iris-versicolor": [0, 1, 0],
  "Iris-virginica": [0, 0, 1]
};

/**
 * Parse Iris CSV text into samples
 * Exported for testing purposes
 */
export function parseIrisCSV(text: string): IrisSample[] {
  // Parse CSV lines and skip header
  const lines = text.trim().split('\n').filter(line => line.length > 0);

  // Skip first line if it looks like a header (contains non-numeric first column)
  const dataLines = lines[0].includes('sepal') ? lines.slice(1) : lines;

  // Parse each line: sepal_length,sepal_width,petal_length,petal_width,species
  const rawData = dataLines
    .map((line, idx) => {
      const parts = line.split(',');

      // Parse features
      const features = [
        parseFloat(parts[0]), // sepal length
        parseFloat(parts[1]), // sepal width
        parseFloat(parts[2]), // petal length
        parseFloat(parts[3])  // petal width
      ];

      // Check for NaN values and log if found
      if (features.some(f => isNaN(f))) {
        console.warn(`⚠️  Line ${idx + 1} has NaN values:`, line);
        console.warn(`   Parts:`, parts);
        console.warn(`   Parsed:`, features);
        return null; // Mark for filtering
      }

      return {
        features,
        species: parts[4]?.trim() || ''
      };
    })
    .filter(item => item !== null) as Array<{ features: number[], species: string }>;

  // Validate we have data
  if (rawData.length === 0) {
    throw new Error('No valid data found in CSV');
  }

  // Find min/max for normalization
  const mins = [Infinity, Infinity, Infinity, Infinity];
  const maxs = [-Infinity, -Infinity, -Infinity, -Infinity];

  for (const sample of rawData) {
    for (let i = 0; i < 4; i++) {
      mins[i] = Math.min(mins[i], sample.features[i]);
      maxs[i] = Math.max(maxs[i], sample.features[i]);
    }
  }
  // Debug: log normalization ranges
  console.log('📊 Normalization ranges:');
  console.log(`  Feature 0: [${mins[0].toFixed(2)}, ${maxs[0].toFixed(2)}]`);
  console.log(`  Feature 1: [${mins[1].toFixed(2)}, ${maxs[1].toFixed(2)}]`);
  console.log(`  Feature 2: [${mins[2].toFixed(2)}, ${maxs[2].toFixed(2)}]`);
  console.log(`  Feature 3: [${mins[3].toFixed(2)}, ${maxs[3].toFixed(2)}]`);

  // Normalize and convert to IrisSample
  const samples: IrisSample[] = rawData.map(sample => {
    // Normalize features to [0, 1]
    const normalized = sample.features.map((val, i) => {
      const range = maxs[i] - mins[i];
      // Handle edge case where all values are the same
      if (range === 0) return 0;
      return (val - mins[i]) / range;
    });

    // Convert to Value objects
    const inputs = normalized.map(n => new Value(n));

    // One-hot encode the label
    const oneHot = speciesMap[sample.species] || [0, 0, 0];
    const outputs = oneHot.map(v => new Value(v));

    return {
      inputs,
      outputs,
      label: sample.species
    };
  });

  return samples;
}

/**
 * Load and preprocess Iris dataset from /data.csv
 */
export async function loadIrisData(): Promise<IrisSample[]> {
  // Fetch CSV from public folder
  const response = await fetch('/data.csv');
  const text = await response.text();

  return parseIrisCSV(text);
}

/**
 * Shuffle array in place
 */
export function shuffle<T>(array: T[]): T[] {
  const shuffled = [...array];
  for (let i = shuffled.length - 1; i > 0; i--) {
    const j = Math.floor(Math.random() * (i + 1));
    [shuffled[i], shuffled[j]] = [shuffled[j], shuffled[i]];
  }
  return shuffled;
}

/**
 * Split data into train and test sets
 */
export function splitTrainTest(
  samples: IrisSample[],
  trainRatio: number = 0.8
): { train: IrisSample[], test: IrisSample[] } {
  const splitIndex = Math.floor(samples.length * trainRatio);
  return {
    train: samples.slice(0, splitIndex),
    test: samples.slice(splitIndex)
  };
}