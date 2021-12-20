<template>
  <div class="container">
    <div class="container-avatar">
      <div class="face-wrapper">
        <div class="face-zoom" :style="{ transform: `scale(${coords.scale})` }">
          <img
            :src="`data:image/png;base64, ${source}`"
            class="face"
            :style="{
              top: coords.top,
              left: coords.left,
            }"
            alt="Face"
          />
        </div>
      </div>
    </div>
  </div>
</template>

<script lang="ts">
import { computed, defineComponent, PropType } from "vue";

type CustomBox = {
  x: number;
  y: number;
  w: number;
  h: number;
};

type Coords = {
  top: string;
  left: string;
  scale: string;
};

export default defineComponent({
  name: "WarningCard",
  props: {
    source: {
      type: String,
      required: true,
    },
    rect: {
      type: Object as PropType<number[]>,
      required: true,
    },
  },
  setup(props) {
    const box = computed<CustomBox>(() => {
      return {
        y: props.rect[0],
        x: props.rect[1],
        h: props.rect[2] - props.rect[0],
        w: props.rect[3] - props.rect[1],
      };
    });


    const coords = computed<Coords>(() => {
      return {
        top: `-${box.value.x - 150 + box.value.h / 2}px`,
        left: `-${box.value.y - 150 + box.value.w / 2}px`,
        scale: `${400 / (box.value.w * 2)}`, //${250 / box.value.w}
      };
    });

    return {
      box,
      coords,
    };
  },
});
</script>

<style scoped lang="scss">
.container {
  width: 300px;
  height: 300px;
  box-sizing: border-box;
  color: white;
  user-select: none;
  position: relative;
  overflow: hidden;
}

.container-avatar {
  width: 300px;
  height: 300px;
  border-radius: 50%;
  box-sizing: border-box;
  position: relative;
  background: transparent;
  border: 11px dashed #e58511;
  overflow: hidden;
}

.face-wrapper {
  clip-path: circle(50% at 50% 50%);
  position: absolute;
  top: -5px;
  left: 0;
  width: 300px;
  height: 300px;
  z-index: 1;
  box-sizing: border-box;
}

.face-zoom {
  width: 300px;
  height: 300px;
}

.face {
  position: absolute;
  top: 0;
  left: 0;
  z-index: 1;
}
</style>
