<template>
  <div class="home">
    <div v-if="loading" class="loading">Загрузка данных ...</div>

    <div v-if="!loading" class="cards-container">
      <card
        v-for="(card, index) in cards"
        :key="card.id"
        :source="source"
        :rect="card.coords"
        :date="getDate(card.datetime)"
        :rating="Math.floor(card.rating)"
        :class="{
          'card--default': true,
          next: index === next,
          current: index === current,
          prev: index === prev,
        }"
      />
    </div>
    <div v-if="!loading" class="text">
      <div class="line">Их разыскивает Дед Мазай!!!</div>
    </div>
  </div>
</template>

<script lang="ts">
import { computed, defineComponent, ref } from "vue";
import Card from "../components/Card.vue";
import moment from "moment";
import { useWebSocket } from "@vueuse/core";

export default defineComponent({
  name: "Home",
  components: {
    Card,
  },
  setup() {
    const cards = ref([]);

    const source = ref("");
    const loading = ref(true);

    useWebSocket(`ws://${window.location.host}${window.location.pathname}ws`, {
      onMessage(ws, e) {
        cards.value = e.data.clients;
        source.value = e.data.suorce_image_base_64;
      },
    });

    const getData = () => {
      fetch("/detect")
        .then((res) => res.json())
        .then((resp) => {
          cards.value = resp.data.data.clients;
          source.value = resp.data.data.suorce_image_base_64;
        });
    };

    setTimeout(() => {
      getData();
      loading.value = false;
    }, 2000);

    const current = ref(0);
    const prev = computed(() => {
      if (current.value === 0) return cards.value.length - 1;
      return current.value - 1;
    });

    const next = computed(() => {
      if (current.value === cards.value.length - 1) return 0;
      return current.value + 1;
    });

    const getDate = (datetime: string) => {
      return moment(datetime, "MM/DD/YYYY, h:mm a").format("DD.MM.YYYY");
    };

    setInterval(() => {
      current.value++;
      if (current.value === cards.value.length) {
        current.value = 0;
      }
    }, 4000);

    return {
      cards,
      current,
      prev,
      next,
      source,
      getDate,
      loading,
    };
  },
});
</script>

<style scoped lang="scss">
.loading {
  font-size: 80px;
  box-sizing: border-box;
  color: #fff;
  height: 200px;
  width: 100%;
  line-height: 200px;
  overflow: hidden;
  font-weight: bolder;
  user-select: none;
  font-family: Roboto, serif;
  text-shadow: 1px 1px 0 #000;
  text-align: center;
  margin-top: 550px;
}

.card--default {
  opacity: 1;
  position: absolute;
  transition: 500ms;
  left: -50%;
  transform: scale(0);
  top: 50%;

  &.current {
    opacity: 1;
    left: calc(75% - 350px);
    transform: scale(1.2);
  }

  &.next {
    opacity: 1;
    left: calc(50% - 350px);
    top: -50%;
    transform: scale(0);
  }

  &.prev {
    opacity: 1;
    left: calc(25% - 350px);
    transform: scale(1.2);
  }
}

.home {
  padding: 0;
  margin: 0;
  background-image: url("../assets/bg_1.svg");
  background-repeat: repeat-x;
  background-size: 115% 105%;
  height: 100%;
  width: 100%;
  overflow: hidden;
}

.cards-container {
  padding-top: 300px;
  position: relative;
}

.text {
  position: absolute;
  left: 0;
  bottom: 0;
  box-sizing: border-box;
  color: #fff;
  font-size: 100px;
  height: 200px;
  width: 100%;
  line-height: 200px;
  overflow: hidden;
  font-weight: bolder;
  text-align: left;
  background: rgba(#000, 0.1);
  user-select: none;
  font-family: Roboto, serif;
  z-index: 99999;


}

.line {
  text-shadow: 3px 1px 0 #000;
  right: -100%;
  top: 0;
  position: absolute;
  animation: RunningLine infinite 12s linear;
  text-transform: uppercase;
}

@keyframes RunningLine {
  from {
    left: 100%;
  }
  to {
    left: -100%;
  }
}
</style>
