<template>
  <div class="wrapper">
    <div class="container-warning">
      <div class="cards">
        <warning-card
          v-for="(card, index) in cards"
          :key="card.id"
          :source="source"
          :rect="card.coords"
          :class="{
            'card--default': true,
            next: index === next,
            current: index === current,
            prev: index === prev,
          }"
        />
      </div>
      <div class="text">Не забудьте оплатить проезд!</div>
    </div>
  </div>
</template>

<script lang="ts">
import { defineComponent, ref } from "vue";
import axios from "axios";
import moment from "moment";
import WarningCard from "@/components/WarningCard.vue";
import { useWebSocket } from "@vueuse/core";

export default defineComponent({
  name: "Warning",
  components: { WarningCard },
  setup() {
    const cards = ref([]);

    const source = ref("");

    const getData = () => {
      fetch("/detect")
        .then((res) => res.json())
        .then((res) => {
        cards.value = res.data.data.clients.filter(
          (client: any) => Math.floor(client.rating) === 1
        );
        source.value = res.data.data.source_image_base_64;
      });
    };

    useWebSocket(`ws://${window.location.host}${window.location.pathname}ws`, {
      onMessage(ws, e) {
        cards.value = e.data.clients;
        source.value = e.data.suorce_image_base_64;
      },
    });

    getData();

    const getDate = (datetime: string) => {
      return moment(datetime, "MM/DD/YYYY, h:mm a").format("DD.MM.YYYY");
    };

    return {
      cards,
      source,
      getDate,
    };
  },
});
</script>

<style scoped lang="scss">
.wrapper {
  height: 100%;
  width: 100%;
  position: relative;
  background: repeating-linear-gradient(
    40deg,
    black 0%,
    yellow 1%,
    yellow 5%,
    black 6%,
    black 9%
  );
}

.container-warning {
  height: 90%;
  width: 90%;
  position: absolute;
  top: 5%;
  left: 5%;
  background: #f4f4f4;
  border-radius: 40px;
  border: #dbdbdb 10px solid;
  box-sizing: border-box;
  display: grid;
  grid-template-areas:
    "cards"
    "text";
  grid-template-rows: 400px 1fr;
  grid-template-columns: 1fr;
}

.text {
  grid-area: text;
  text-align: center;
  font-size: 130px;
  font-weight: bold;
  height: 100%;
  width: 100%;
  padding-top: 80px;
}

.cards {
  grid-area: cards;
  display: flex;
  justify-content: center;
  align-items: center;
}

.card--default {
  margin: 10px;
}
</style>
