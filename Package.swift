// swift-tools-version:4.0

import PackageDescription

let package = Package(
    name: "NeuralNet-MNIST",
    products: [
        .library(
            name: "NeuralNet-MNIST",
            targets: ["NnMnist"]),
    ],
    dependencies: [
    ],
    targets: [
        .target(
            name: "NnMnist",
            dependencies: []),
    ]
)
