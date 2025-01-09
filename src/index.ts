#!/usr/bin/env node
import { MetalExpertServer } from './server.js';

async function main() {
  const server = new MetalExpertServer();
  await server.run();
}

main().catch((error) => {
  console.error('Server error:', error);
  process.exit(1);
});
