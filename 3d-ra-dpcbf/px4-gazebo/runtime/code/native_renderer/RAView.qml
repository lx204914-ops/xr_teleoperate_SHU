import QtQuick 2.12
Rectangle {
  width: 640; height: 42; radius: 5
  color: "#EFFFFFFF"; border.color: "#BBC8D2"
  Text { anchors.centerIn: parent; text: RAView.status; font.pixelSize: 17; color: "#243746" }
}
