<template>
  <div id="nav">
    <router-link to="/">Home</router-link>
    <router-link to="/schema">Schema</router-link>
    <router-link to="/warning">Warning</router-link>
  </div>
  <router-view />
</template>

<style lang="scss">
html, body, #app {
  padding: 0;
  margin: 0;
  height: 100%;
  width: 100%;
}

#app {
  font-family: Avenir, Helvetica, Arial, sans-serif;
  -webkit-font-smoothing: antialiased;
  -moz-osx-font-smoothing: grayscale;
  text-align: center;
  color: #2c3e50;
}

#nav {
  padding: 5px;
  position: absolute;
  display: flex;
  justify-content: flex-start;
  align-items: center;
  z-index: 999;
  left: 25px;
  top: 25px;
  width: auto;
  height: auto;
  background: whitesmoke;
  border-radius: 30px;
  box-shadow: 0 3px 0 0 rgba(#000, .15);
  opacity: 0;
  transition: 250ms;

  &:hover {
    opacity: 1;
  }

  a {
    margin: 5px 10px;
    font-weight: bold;
    color: #2c3e50;

    &.router-link-exact-active {
      color: #42b983;
    }
  }
}
</style>
