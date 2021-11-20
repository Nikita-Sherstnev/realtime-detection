module.exports = {
  devServer: {
    proxy: {
      "/": {
        target: "http://localhost:9000",
        changeOrigin: true,
        ws: true,
        logLevel: "debug",
      },
    },
  },
};
