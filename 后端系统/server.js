// server.js
const express = require('express');
const cors = require('cors');
const http = require('http');
const userSkillRoutes = require('./routes/userSkillRoutes');
require('dotenv').config();

const app = express();

const PORT = process.env.PORT || 3000;

app.use(cors());
app.use(express.json());
const WebSocket = require('ws');
const server = http.createServer(app);
const wss = new WebSocket.Server({ server });
// 模拟数据库：存储每个房间/路径的状态
const operationStatus = new Map(); // roomId -> { completed: true/false, data: {}, timestamp }
// 存储所有活跃的 WebSocket 客户端（前端 B）
const clients = new Set();

// WebSocket 连接处理（前端 B 连接进来）
wss.on('connection', (ws) => {
  console.log('前端 B 已连接');
  clients.add(ws);

  ws.send(JSON.stringify({ type: 'info', message: '已连接到服务器' }));

  // 客户端断开时移除
  ws.on('close', () => {
    console.log('前端 B 断开连接');
    clients.delete(ws);
  });

  ws.on('error', (err) => {
    console.error('WebSocket 错误:', err);
    clients.delete(ws);
  });
});



// 接口 1: 前端 A 调用 - 标记操作完成
app.get('/api/operation-complete/:roomId', (req, res) => {
  const { roomId } = req.params;
  // const { path_id } = req.body;

  operationStatus.set(roomId, {
    completed: true,
    timestamp: new Date().toISOString()
  });

  // 广播给所有连接的前端 B
  const message = JSON.stringify({ type: 'path_id', roomId, timestamp: new Date().toISOString() });
  clients.forEach(client => {
    if (client.readyState === WebSocket.OPEN) {
      client.send(message);
      console.log(`✅ 广播给前端B，房间: ${roomId}`);

    }
  });
  console.log(`✅ 前端A完成操作，房间: ${roomId}`);
  res.json({ success: true, message: '状态已更新' });
});

// 接口 1: 前端 A 调用 - 标记操作完成
app.get('/api/questions-complete', (req, res) => {

  // 广播给所有连接的前端 B
  const message = JSON.stringify({ type: 'begin_questions', timestamp: new Date().toISOString() });

  clients.forEach(client => {
    if (client.readyState === WebSocket.OPEN) {
      client.send(message);
      console.log(`✅ 广播给做题页面`);

    }
  });
  console.log(`✅ 前端A完成操作`);
  res.json({ success: true, message: '状态已更新' });
});



app.use('/api', userSkillRoutes);

server.listen(PORT, '0.0.0.0', () => {
  console.log(`Server is running on http://localhost:${PORT}`);
});