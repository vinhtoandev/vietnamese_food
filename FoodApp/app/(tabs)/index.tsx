import React from "react";
import { View, Text, StyleSheet, FlatList, Image, Dimensions } from "react-native";
import Banner from "../../components/Banner/Banner";

const { width } = Dimensions.get("window");

const foods = [
  { id: "1", name: "Phở Bò", price: "50.000đ", image: require("../../assets/banners/banhmi.webp") },
  { id: "2", name: "Bún Chả", price: "45.000đ", image: require("../../assets/banners/bunbohue.jpg") },
  { id: "3", name: "Bánh Mì", price: "20.000đ", image: require("../../assets/banners/cafe.jpg") },
  { id: "4", name: "Cơm Tấm", price: "40.000đ", image: require("../../assets/banners/banhmi.webp") },
  { id: "5", name: "Gỏi Cuốn", price: "30.000đ", image: require("../../assets/banners/banhmi.webp") },
  { id: "6", name: "Bánh Xèo", price: "35.000đ", image: require("../../assets/banners/banhmi.webp") },
  { id: "7", name: "Mì Quảng", price: "45.000đ", image: require("../../assets/banners/banhmi.webp") },
  { id: "8", name: "Chả Giò", price: "25.000đ", image: require("../../assets/banners/banhmi.webp") },
  { id: "9", name: "Lẩu Thái", price: "120.000đ", image: require("../../assets/banners/banhmi.webp") },
];

export default function HomeScreen() {
  const renderFoodItem = ({ item }: any) => (
    <View style={styles.card}>
      <Image source={item.image} style={styles.foodImage} />
      <Text style={styles.foodName}>{item.name}</Text>
      <Text style={styles.foodPrice}>{item.price}</Text>
    </View>
  );

  return (
    <View style={styles.container}>
      {/* Header */}
      <View style={styles.header}>
        <Text style={styles.headerText}>Vietnamese Food App</Text>
      </View>

      {/* Banner */}
      <Banner />

      {/* Food List */}
      <Text style={styles.sectionTitle}>Danh sách món ăn</Text>
      <FlatList
        data={foods}
        renderItem={renderFoodItem}
        keyExtractor={(item) => item.id}
        numColumns={2}
        contentContainerStyle={styles.foodList}
      />
    </View>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: "#fff",
  },
  header: {
    paddingTop: 50,
    paddingBottom: 15,
    backgroundColor: "#FF5722",
    alignItems: "center",
    justifyContent: "center",
  },
  headerText: {
    fontSize: 20,
    fontWeight: "bold",
    color: "#fff",
  },
  sectionTitle: {
    fontSize: 18,
    fontWeight: "bold",
    marginVertical: 10,
    marginLeft: 10,
  },
  foodList: {
    paddingHorizontal: 8,
  },
  card: {
    flex: 1,
    margin: 8,
    backgroundColor: "#f9f9f9",
    borderRadius: 12,
    padding: 10,
    alignItems: "center",
    shadowColor: "#000",
    shadowOpacity: 0.1,
    shadowOffset: { width: 0, height: 2 },
    shadowRadius: 5,
    elevation: 3,
  },
  foodImage: {
    width: width / 2 - 40,
    height: 120,
    borderRadius: 10,
  },
  foodName: {
    fontSize: 16,
    fontWeight: "600",
    marginTop: 8,
  },
  foodPrice: {
    fontSize: 14,
    color: "#FF5722",
    marginTop: 4,
  },
});
