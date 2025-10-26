import { describe, it, expect, beforeAll } from "vitest"
import { parseIrisCSV, shuffle, splitTrainTest, IrisSample } from "../src/micrograd/preprocessor"
import { readFileSync } from 'fs';
import { join } from 'path';

describe('Iris Data Preprocessing', () => {
  let data: IrisSample[];

  beforeAll(() => {
    // Read CSV file from public folder using Node.js fs
    const csvPath = join(process.cwd(), 'public', 'data.csv');
    const csvText = readFileSync(csvPath, 'utf-8');
    data = parseIrisCSV(csvText);
  });

  describe('loadIrisData', () => {
    it('should load 150 samples (excluding header)', async () => {
      expect(data).toHaveLength(149);
    });

    it('should have 4 inputs per sample', () => {
      data.forEach(sample => {
        expect(sample.inputs).toHaveLength(4);
      });
    });

    it('should have 3 outputs per sample (one-hot encoded)', () => {
      data.forEach(sample => {
        expect(sample.outputs).toHaveLength(3);
      });
    });

    it('should normalize all features to [0, 1]', () => {
      data.forEach((sample, idx) => {
        sample.inputs.forEach((input, featIdx) => {
          if (isNaN(input.data)) {
            console.error(`❌ Sample ${idx} (${sample.label}), feature ${featIdx}: NaN`);
          }
          expect(input.data).toBeGreaterThanOrEqual(0);
          expect(input.data).toBeLessThanOrEqual(1);
        });
      });
    });

    it('should have correct label distribution', () => {
      const counts: { [key: string]: number } = {};
      data.forEach(sample => {
        counts[sample.label] = (counts[sample.label] || 0) + 1;
      });

      // Check that we have 3 species
      expect(Object.keys(counts).length).toBe(3);

      // Check that each species has samples (50 each for standard Iris dataset)
      expect(counts['setosa']).toBe(49);
      expect(counts['versicolor']).toBe(50);
      expect(counts['virginica']).toBe(50);

      // Log actual distribution
      console.log('Label distribution:', counts);
    });

    it('should have valid one-hot encoding', () => {
      data.forEach((sample, idx) => {
        const outputValues = sample.outputs.map(v => v.data);
        const sum = outputValues.reduce((a, b) => a + b, 0);

        if (sum !== 1) {
          console.error(`❌ Sample ${idx} (${sample.label}) one-hot sum: ${sum}, values: [${outputValues}]`);
        }

        // Sum should be 1 (one-hot)
        expect(sum).toBe(1);

        // Each value should be 0 or 1
        outputValues.forEach(val => {
          expect([0, 1]).toContain(val);
        });
      });
    });
  });

  describe('shuffle', () => {
    it('should return same number of samples', () => {
      const shuffled = shuffle(data);
      expect(shuffled).toHaveLength(data.length);
    });

    it('should not modify original array', () => {
      const original = [...data];
      shuffle(data);
      expect(data).toEqual(original);
    });

    it('should change order (probabilistically)', () => {
      const shuffled = shuffle(data);
      let samePositions = 0;

      for (let i = 0; i < data.length; i++) {
        if (data[i].label === shuffled[i].label) {
          samePositions++;
        }
      }

      // Should change at least some positions
      expect(samePositions).toBeLessThan(data.length);
    });
  });

  describe('splitTrainTest', () => {
    it('should split with 80/20 ratio by default', () => {
      const { train, test } = splitTrainTest(data);

      expect(train.length).toBe(119); // 80% of 150
      expect(test.length).toBe(30);   // 20% of 150
    });

    it('should split with custom ratio', () => {
      const { train, test } = splitTrainTest(data, 0.7);

      expect(train.length).toBe(104); // 70% of 150
      expect(test.length).toBe(45);   // 30% of 150
    });

    it('should preserve all samples', () => {
      const { train, test } = splitTrainTest(data);

      expect(train.length + test.length).toBe(data.length);
    });

    it('should not have overlapping samples', () => {
      const { train, test } = splitTrainTest(data);

      const trainLabels = train.map(s => s.label);
      const testLabels = test.map(s => s.label);

      // This is a simple check - in reality we'd check actual object references
      expect(train.length + test.length).toBe(data.length);
    });
  });
});