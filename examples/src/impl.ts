
import {
  Tensor, Tensor2D, Tensor3D, browser, GraphModel, loadGraphModel, tidy,
  expandDims, transpose, unstack, topk, div, maximum
} from '@tensorflow/tfjs';

export class Detect {
  private model?: GraphModel;
  private modelUrl: string;
  private isInitialized: boolean;

  constructor(url: string) {
    this.modelUrl = url;
    this.isInitialized = false;

    console.log(`Start loading ${url}`);
    loadGraphModel(url).then((model) => {
      console.log('Success to load model.')
      this.model = model;
      this.isInitialized = true;
    }).catch(err => {
      console.log(`Failed to load model, ${err}`);
    });
  }


  detect(img: ImageData) {
    if (!this.model) {
      return [null, null];
    }

    const height = img.height;
    const width = img.width;
    const resized_height = 128;
    const resized_width = 128;
    const height_ratio = resized_height / height;
    const width_ratio = resized_width / width;

    const batched = tidy(() => {
      let img_tensor = browser.fromPixels(img);
      img_tensor = img_tensor.resizeBilinear([resized_height, resized_width]);
      img_tensor = div(img_tensor, 255);
      img_tensor = transpose(img_tensor, [2, 0, 1]);
      return expandDims(img_tensor);
    });

    const [regs, heatmap] = this.model.execute(batched) as Tensor[];
    const [prob, result] = this.decode(heatmap);
    let r = result as number[][];
    if (result) {
      // Resized to initial image size.
      for (let i = 0; i < r.length; i++) {
        r[i][0] = r[i][0] / width_ratio;
        r[i][1] = r[i][1] / height_ratio;
      }
    }
    return [heatmap, result, prob];
  }

  decode(heatmap: Tensor, iterationLimit = 500) {
    const K = 5;

    let pred_ls: number[][] = [[], [], [], []];
    let pos_ls: number[][][] = [[], [], [], []];

    let [batch, channel, height, width] = heatmap.shape;
    heatmap = heatmap.reshape([channel, height, width]);

    let channels = unstack(heatmap, 0);
    for (let i = 0; i < channels.length; i++) {
      let c = channels[i].reshape([-1]);
      let { values, indices } = topk(c, K, true);

      let idx_array = indices.arraySync() as number[];
      let pred_array = values.arraySync() as number[];

      for (let idx of idx_array) {
        let w = idx % width;
        let h = Math.floor(idx / width);
        pos_ls[i].push([w, h]);
      }
      for (let pred of pred_array) {
        pred_ls[i].push(pred);
      }
    }

    let initial = [0, 0, 0, 0];
    let queue: number[][] = [];
    let visit = new Set();
    queue.push(initial);

    let count = 0;
    while (queue.length > 0) {
      let top = queue.shift();

      // Error and visit check.
      if (!top) {
        throw new Error('Failed to obtain top of queue.');
      }

      // Check already visit.
      let key = top.join('-');
      if (visit.has(key)) {
        continue;
      }
      visit.add(key);

      // Check max iteration limit.
      if (count > iterationLimit) {
        break;
      }
      count++;

      let violation = false;
      let corners: number[][] = [];
      let pred_sum = 0;
      for (let i = 0; i < 4; i++) {
        let n = top[i];
        if (top[i] >= K) {
          violation = violation || true;
        } else {
          corners.push(pos_ls[i][n]);
          pred_sum += pred_ls[i][n];
        }
      }
      if (violation) {
        continue;
      }

      // Check corners can construct a valid rectangle.
      if (this.isValidRectangle(corners)) {
        return [pred_sum/4, corners];
      } else {
        queue.push([top[0] + 1, top[1], top[2], top[3]]);
        queue.push([top[0], top[1] + 1, top[2], top[3]]);
        queue.push([top[0], top[1], top[2] + 1, top[3]]);
        queue.push([top[0], top[1], top[2], top[3] + 1]);
      }
    }
    return [0, null];
  }



  /**
   * Check if corne
   * @param corners List of [x, y] pairs.
   */
  isValidRectangle(corners: number[][]): boolean {
    let result = true;
    for (let i1 = 0; i1 < corners.length; i1++) {
      let i2 = (i1 + 1) % corners.length;
      let i3 = (i2 + 1) % corners.length;
      let v1x = corners[i2][0] - corners[i1][0];
      let v1y = corners[i2][1] - corners[i1][1];
      let v2x = corners[i3][0] - corners[i2][0];
      let v2y = corners[i3][1] - corners[i2][1];
      let cross = v1x * v2y - v1y * v2x;
      result = result && (cross < 0);
    }
    return result;
  }
}

export class WebcamCapture {
  /**
   * Wrapper for  
   */

  private stream?: MediaStream;
  private video: HTMLVideoElement;
  private context: CanvasRenderingContext2D | null;
  private width: number;
  private height: number;

  constructor(width: number, height: number) {
    this.width = width;
    this.height = height;

    this.context = document.createElement('canvas').getContext('2d');
    this.video = document.createElement("video");
    this.video.width = width;
    this.video.height = height;
  }

  start(isFront: boolean = false) {
    if (navigator.mediaDevices.getUserMedia) {
      const constrain = {
        video: {
          facingMode: isFront ? "user" : "environment",
          width: this.width,
          height: this.height,
        }
      }
      navigator.mediaDevices.getUserMedia(constrain).then(stream => {
        this.video.srcObject = stream;
        this.video.play();
      });
    }
  }

  stop() {
  }

  capture(): ImageData {
    if (!this.context) {
      throw new Error('Failed to obtain 2d context.')
    }
    this.context.canvas.width = this.width;
    this.context.canvas.height = this.height;
    this.context.drawImage(this.video, 0, 0, this.width, this.height)
    let vals = this.context.getImageData(0, 0, this.width, this.height);
    return vals
  }
}

export function heatmapImage(heatmap: Tensor, canvas: HTMLCanvasElement) {
  let channels = unstack(heatmap, 0);

  let result = null;
  for (let c of channels) {
    if (!result) {
      result = c;
    } else {
      result = maximum(result, c);
    }
  }
  if (!result) {
    throw new Error('Failed to generate heatmap tensor.');
  }

  result = transpose(result, [1, 2, 0]);
  result = result.resizeBilinear([canvas.height, canvas.width]) as Tensor3D;
  console.log(result?.shape)
  browser.toPixels(result, canvas)
  return result?.array()
}
