<template>
  <div class="container">
    <div class="status-bar">
      <div class="stars">
        <star v-for="i in rating" :key="i" />
      </div>
      <div class="date">{{ date }}</div>
    </div>
    <div class="container-avatar">
      <img src="../assets/rabbit.png" alt="rabbit mask" class="rabbit" />

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
import Star from "@/components/Star.vue";

type RenderBox = {
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
  name: "Card",
  components: { Star },
  props: {
    source: {
      type: String,
      required: true,
    },
    rect: {
      type: Object as PropType<number[]>,
      required: true,
    },
    date: {
      type: String,
      required: true,
    },
    rating: {
      type: Number,
      required: true,
    },
  },
  setup(props) {
    const box = computed<RenderBox>(() => {
      return {
        y: props.rect[0],
        x: props.rect[1],
        h: props.rect[2] - props.rect[0],
        w: props.rect[3] - props.rect[1],
      };
    });

    const coords = computed<Coords>(() => {
      const yFactor = box.value.y > 215 ? 1 : -1;

      return {
        top: `-${box.value.x - 45 + box.value.h / 2}px`,
        left: `${
          (Math.abs(box.value.y - 215) + (box.value.w / 2) * yFactor) * -yFactor
        }px`,
        scale: `${400 / (box.value.w * 2)}`,
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
  width: 700px;
  height: auto;
  box-sizing: border-box;
  color: white;
  user-select: none;
  position: relative;
}

.status-bar {
  position: absolute;
  z-index: 1000;
  top: 0;
  left: calc(50% - 150px);
  width: 300px;
  height: auto;
}

.date {
  text-shadow: 1px 2px 0 #000;
  font-size: 32px;
  border-radius: 0 0 30px 30px;
  border: 3px solid #ff9f04;
  width: 65%;
  margin: 0 auto;
  background: rgba(#000, 0.35);
  border-top: none;
}

.stars {
  border-radius: 25px;
  border: 3px solid #ff9f04;
  background: rgba(#fff, 0.65);
}

.container-avatar {
  width: 100%;
  height: auto;
  border-radius: 50%;
  box-sizing: border-box;
  position: relative;
  background: transparent;
}

.rabbit {
  position: absolute;
  top: -280px;
  left: calc(50% - 350px);
  height: auto;
  width: 100%;
  z-index: 100;
}

.face-wrapper {
  clip-path: circle(50% at 50% 50%);
  position: absolute;
  top: 280px;
  left: 105px;
  width: 480px;
  height: 90px;
  z-index: 1;
}

.face-zoom {
  clip-path: circle(50% at 50% 50%);
  width: 400px;
  height: 90px;
}

.face {
  position: absolute;
  z-index: 1;
}
</style>
