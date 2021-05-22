

import * as impl from './impl';

const globalObj = ((typeof window !== 'undefined') ? window : global) as any;
globalObj.yaoyorozu = impl

declare global {
  const webcam: impl.WebcamCapture;
}