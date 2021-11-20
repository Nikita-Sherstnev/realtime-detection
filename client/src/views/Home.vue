<template>
  <div class="rt-home">
    <img :src="`data:image/jpg;base64,${img}`" alt="stream" />
  </div>
</template>

<script lang="ts">
import { defineComponent, ref } from "vue";
import { useWebSocket } from "@vueuse/core";

export default defineComponent({
  name: "Home",
  components: {},
  setup() {
    const img = ref("");

    useWebSocket(`ws://${window.location.host}${window.location.pathname}ws`, {
      onConnected(ws) {
        ws.send("1");
      },
      onMessage(ws, e) {
        img.value = e.data;
        ws.send("1");
      },
      onError(ws, e) {
        console.log(e);
        ws.send("1");
      },
    });

    return {
      img,
    };
  },
});
</script>
