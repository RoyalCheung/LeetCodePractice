public class KIC {
    private class item{
        private String name;
        private int key;
        item(String name, int key){
            this.name = name;
            this.key = key;
        }
        public int key(){
            return this.key;
        }
    }
    private int R;
    private item[] a;
    private int[] count;
    private int p;
    private int N;
    private item[] aux;
    public KIC(int R, int N){
        this.R = R;
        this.N = N;
        a = new item[N];
        p = 0;
    }
    public void add(String name, int key){
        a[p++] = new item(name, key);
    }
    private void computeFreqCounts(){
        for (int i = 0; i < N; i++){
            if (a[i] != null){
                count[a[i].key()+1]++;
            }
        }
    }
    private void transformC2I(){
        for (int r = 0; r < R; r++){
            count[r+1] += count[r];
        }
    }
    private void distribute(){
        aux = new item[N];
        count = new int[R+1];
        computeFreqCounts();
        transformC2I();
        for (int i = 0; i < N; i++){
            aux[count[a[i].key()]++] = a[i];
        }
    }
    private void copyback(){
        distribute();
        for (int i = 0; i < N; i++){
            a[i] = aux[i];
        }
    }
    private void keyIndexCount(){
        aux = new item[N];
        count = new int[R+1];
        for (int i = 0; i < N; i++){
            count[a[i].key()+1]++;
        }
        for (int r = 0; r < R; r++){
            count[r+1] += count[r];
        }
        for (int i = 0; i < N; i ++){
            aux[count[a[i].key()]++] = a[i];
        }
        for (int i = 0; i < N; i++){
            a[i] = aux[i];
        }
    }
    public static void main(String[] args) {

    }
}
