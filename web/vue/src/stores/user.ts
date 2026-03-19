import { reactive, readonly } from 'vue';

interface UserState {
  isLoggedIn: boolean;
  id: string;
  username: string;
  nickname: string;
  email: string;
  avatar: string;
  historyCount: number;
  disputesCount: number;
  accuracy: string;
}

const STORAGE_KEY = 'lare_user';

function loadFromStorage(): UserState {
  try {
    const saved = localStorage.getItem(STORAGE_KEY);
    if (saved) return JSON.parse(saved);
  } catch { /* ignore */ }
  return getDefaultState();
}

function getDefaultState(): UserState {
  return {
    isLoggedIn: false,
    id: '', username: '', nickname: '', email: '', avatar: '',
    historyCount: 0, disputesCount: 0, accuracy: '0%'
  };
}

const state = reactive<UserState>(loadFromStorage());

function save() {
  localStorage.setItem(STORAGE_KEY, JSON.stringify(state));
}

export function useUserStore() {
  function loginUser(userData: Partial<UserState>) {
    Object.assign(state, userData, { isLoggedIn: true });
    save();
  }

  function logoutUser() {
    Object.assign(state, getDefaultState());
    localStorage.removeItem(STORAGE_KEY);
  }

  function updateUser(partial: Partial<UserState>) {
    Object.assign(state, partial);
    save();
  }

  return {
    state: readonly(state) as UserState,
    raw: state,
    loginUser,
    logoutUser,
    updateUser,
  };
}
