import { createApp } from "vue";
import App from "./App.vue";
import router from "./router";
import installElementPlus from "./plugins/element";

import "@fortawesome/fontawesome-free/css/all.min.css";
import "./styles/gilroy.scss";
import "./styles/srollbar.scss";

const app = createApp(App);

installElementPlus(app);

app.use(router).mount("#app");
