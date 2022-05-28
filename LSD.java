public class LSD {
    public static void sort(String[] a, int w){
        int R = 256;
        String[] aux = new String[a.length];
        for (int i = w-1; i >= 0; i--) {
            int[] count = new int[R + 1];
            for (int r = 0; r < a.length; r++) {
                count[a[r].charAt(i) + 1]++;
            }
            for (int r = 0; r < R; r++){
                count[r+1] += count[r];
            }
            for (int r = 0; r < a.length; r++){
                aux[count[a[r].charAt(i)]++] = a[r];
            }
            for (int r = 0; r < a.length; r++){
                a[r] = aux[r];
            }
        }
    }

    public static void main(String[] args) {
        String[] temp = new String[2];
        temp[0] = "4PGC938";
        temp[1] = "2IYE230";
        LSD.sort(temp, 5);
        for (String tmepp: temp){
            System.out.println(tmepp);
        }
    }
}
