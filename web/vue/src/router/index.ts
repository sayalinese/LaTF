import { createRouter, createWebHistory, RouteRecordRaw } from 'vue-router';
import DetectionView from '../views/DetectionView.vue';
import SessionDetectionView from '../views/SessionDetectionView.vue';
import PluginTestView from '../views/PluginTestView.vue';
import ProfileView from '../views/ProfileView.vue';

const routes: Array<RouteRecordRaw> = [
  {
    path: '/',
    name: 'Detection',
    component: DetectionView,
  },
  {
    path: '/session',
    name: 'Session',
    component: SessionDetectionView,
  },
  {
    path: '/plugin',
    name: 'Plugin',
    component: PluginTestView,
  },
  {
    path: '/profile',
    name: 'Profile',
    component: ProfileView,
  }
];

const router = createRouter({
  history: createWebHistory(),
  routes,
});

export default router;