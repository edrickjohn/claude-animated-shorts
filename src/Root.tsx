import React from "react";
import { Composition } from "remotion";
import { BarbieShort, BARBIE_FRAMES } from "../topics/barbie-arms/video/Barbie";
import { MolassesShort, MOLASSES_FRAMES } from "../topics/molasses-flood/video/Molasses";

export const RemotionRoot: React.FC = () => (
  <>
    <Composition id="BarbieShort" component={BarbieShort} durationInFrames={BARBIE_FRAMES} fps={30} width={1080} height={1920} />
    <Composition id="MolassesShort" component={MolassesShort} durationInFrames={MOLASSES_FRAMES} fps={30} width={1080} height={1920} />
  </>
);
