module.exports = {
    css: {
        loaderOptions: {
            less: {
                lessOptions: {
                    javascriptEnabled: true,
                }
            },
        },
    },
    publicPath: "./",
    devServer: {
        port: 8561,
        disableHostCheck: true,
        compress: true,
        host: "0.0.0.0",
        hot: true,
        proxy: {
            '/api/chat': { target: 'http://localhost:8080', changeOrigin: true },
            '/api/files': { target: 'http://localhost:8080', changeOrigin: true },
            '/api/source': { target: 'http://localhost:8080', changeOrigin: true },
            '/api/highlight': { target: 'http://localhost:8080', changeOrigin: true },
            '/api/mermaid': { target: 'http://localhost:8080', changeOrigin: true },
            '/api/umap': { target: 'http://localhost:8080', changeOrigin: true },
            '/api/ast': { target: 'http://localhost:8080', changeOrigin: true },
        }
    },
    configureWebpack: {
        plugins: [
        ]
    }
}